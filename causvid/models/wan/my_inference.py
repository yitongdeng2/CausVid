from causvid.models import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper
)
from typing import List, Optional
import torch


class MyInferencePipeline(torch.nn.Module):
    def __init__(self, generator, text_encoder, vae, args, dtype, device):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = generator.to(device=device, dtype=dtype)
        self.text_encoder = text_encoder.to(device=device, dtype=dtype)
        self.vae = vae.to(device=device, dtype=dtype)

        # Step 2: Initialize all causal hyperparmeters
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device)
        assert self.denoising_step_list[-1] == 0
        # remove the last timestep (which equals zero)
        self.denoising_step_list = self.denoising_step_list[:-1]

        self.scheduler = self.generator.get_scheduler()
        if args.warp_denoising_step:  # Warp the denoising step according to the scheduler time shift
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).cuda()
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.args = args
        self.num_frame_per_block = getattr(
            args, "num_frame_per_block", 1)

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, 32760, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 32760, 12, 128], dtype=dtype, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache = crossattn_cache  # always store the clean cache

    def inference(self, noise: torch.Tensor, text_prompts: List[str], start_latents: Optional[torch.Tensor] = None, return_latents: bool = False) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        output = torch.zeros(
            [batch_size, num_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )

            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False

        num_input_blocks = start_latents.shape[1] // self.num_frame_per_block if start_latents is not None else 0

        # Step 2: Temporal denoising loop
        num_blocks = num_frames // self.num_frame_per_block
        for block_index in range(num_blocks):
            noisy_input = noise[:, block_index *
                                self.num_frame_per_block:(block_index + 1) * self.num_frame_per_block]

            if start_latents is not None and block_index < num_input_blocks:
                timestep = torch.ones(
                    [batch_size, self.num_frame_per_block], device=noise.device, dtype=torch.int64) * 0

                current_ref_latents = start_latents[:, block_index * self.num_frame_per_block:(
                    block_index + 1) * self.num_frame_per_block]
                output[:, block_index * self.num_frame_per_block:(
                    block_index + 1) * self.num_frame_per_block] = current_ref_latents

                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                    current_end=(block_index + 1) *
                    self.num_frame_per_block * self.frame_seq_length
                )
                continue

            # Step 2.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = torch.ones(
                    [batch_size, self.num_frame_per_block], device=noise.device, dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                        current_end=(
                            block_index + 1) * self.num_frame_per_block * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep *
                        torch.ones([batch_size], device="cuda",
                                   dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                        current_end=(
                            block_index + 1) * self.num_frame_per_block * self.frame_seq_length
                    )

            # Step 2.2: rerun with timestep zero to update the cache
            output[:, block_index * self.num_frame_per_block:(
                block_index + 1) * self.num_frame_per_block] = denoised_pred

            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                current_end=(block_index + 1) *
                self.num_frame_per_block * self.frame_seq_length
            )

        # Step 3: Decode the output
        video = self.vae.decode_to_pixel(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        else:
            return video
