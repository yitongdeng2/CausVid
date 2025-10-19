from causvid.models import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper
)
from typing import List, Optional
import torch
import torch.nn.functional as F
from wan.text2video import WanVAE

# # VACE Helpers
# def prepare_source(self, src_video, src_mask, src_ref_images, num_frames, image_size, device):
#     area = image_size[0] * image_size[1]
#     self.vid_proc.set_area(area)
#     if area == 720*1280:
#         self.vid_proc.set_seq_len(75600)
#     elif area == 480*832:
#         self.vid_proc.set_seq_len(32760)
#     else:
#         raise NotImplementedError(f'image_size {image_size} is not supported')

#     image_size = (image_size[1], image_size[0])
#     image_sizes = []
#     for i, (sub_src_video, sub_src_mask) in enumerate(zip(src_video, src_mask)):
#         if sub_src_mask is not None and sub_src_video is not None:
#             src_video[i], src_mask[i], _, _, _ = self.vid_proc.load_video_pair(sub_src_video, sub_src_mask)
#             src_video[i] = src_video[i].to(device)
#             src_mask[i] = src_mask[i].to(device)
#             src_mask[i] = torch.clamp((src_mask[i][:1, :, :, :] + 1) / 2, min=0, max=1)
#             image_sizes.append(src_video[i].shape[2:])
#         elif sub_src_video is None:
#             src_video[i] = torch.zeros((3, num_frames, image_size[0], image_size[1]), device=device)
#             src_mask[i] = torch.ones_like(src_video[i], device=device)
#             image_sizes.append(image_size)
#         else:
#             src_video[i], _, _, _ = self.vid_proc.load_video(sub_src_video)
#             src_video[i] = src_video[i].to(device)
#             src_mask[i] = torch.ones_like(src_video[i], device=device)
#             image_sizes.append(src_video[i].shape[2:])

#     for i, ref_images in enumerate(src_ref_images):
#         if ref_images is not None:
#             image_size = image_sizes[i]
#             for j, ref_img in enumerate(ref_images):
#                 if ref_img is not None:
#                     ref_img = Image.open(ref_img).convert("RGB")
#                     ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
#                     if ref_img.shape[-2:] != image_size:
#                         canvas_height, canvas_width = image_size
#                         ref_height, ref_width = ref_img.shape[-2:]
#                         white_canvas = torch.ones((3, 1, canvas_height, canvas_width), device=device) # [-1, 1]
#                         scale = min(canvas_height / ref_height, canvas_width / ref_width)
#                         new_height = int(ref_height * scale)
#                         new_width = int(ref_width * scale)
#                         resized_image = F.interpolate(ref_img.squeeze(1).unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0).unsqueeze(1)
#                         top = (canvas_height - new_height) // 2
#                         left = (canvas_width - new_width) // 2
#                         white_canvas[:, :, top:top + new_height, left:left + new_width] = resized_image
#                         ref_img = white_canvas
#                     src_ref_images[i][j] = ref_img.to(device)
#     return src_video, src_mask, src_ref_images

# def vace_encode_frames(frames, ref_images, vae, masks=None):
#     if ref_images is None:
#         ref_images = [None] * len(frames)
#     else:
#         assert len(frames) == len(ref_images)

#     if masks is None:
#         latents = vae.encode(frames)
#     else:
#         masks = [torch.where(m > 0.5, 1.0, 0.0) for m in masks]
#         inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
#         reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
#         inactive = vae.encode(inactive)
#         reactive = vae.encode(reactive)
#         latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]

#     cat_latents = []
#     for latent, refs in zip(latents, ref_images):
#         if refs is not None:
#             if masks is None:
#                 ref_latent = vae.encode(refs)
#             else:
#                 ref_latent = vae.encode(refs)
#                 ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
#             assert all([x.shape[1] == 1 for x in ref_latent])
#             latent = torch.cat([*ref_latent, latent], dim=1)
#         cat_latents.append(latent)
#     return cat_latents

# def vace_latent(z, m):
#         return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]
# # Helpers

class MyInferencePipeline(torch.nn.Module):
    def __init__(self, generator, text_encoder, vae, args, dtype, device):
        super().__init__()
        self.dtype=dtype
        self.device=device
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

        # SET VACE CONTEXT
        # this vae might be different from the VAE wrapper, or maybe not, not 100% sure
        z = torch.load("/home/yitong-moonlake/CausVid/prompt_files/extracted_vace_context.pt")
        z = [item.to(device=self.device, dtype=self.dtype) for item in z]
        vace_context = z
        vace_context_scale = 0.0
        # SET VACE CONTEXT

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
                    vace_context=vace_context, vace_context_scale=vace_context_scale,
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
                        vace_context=vace_context, vace_context_scale=vace_context_scale,
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
                        vace_context=vace_context, vace_context_scale=vace_context_scale,
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
                vace_context=vace_context, vace_context_scale=vace_context_scale,
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
