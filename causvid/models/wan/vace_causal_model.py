from diffusers.configuration_utils import register_to_config
from causvid.models.wan.causal_model import CausalWanAttentionBlock, CausalWanModel
import torch.nn as nn
from causvid.models.wan.wan_base.modules.model import WanAttentionBlock, sinusoidal_embedding_1d
import torch

class VaceWanAttentionBlock(WanAttentionBlock):
    def __init__(
            self,
            cross_attn_type,
            dim,
            ffn_dim,
            num_heads,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=False,
            eps=1e-6,
            block_id=0
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            print("self.before_proj: ", self.before_proj)
            print("c shape: ", c.shape)
            print("self.before_proj(c) shape: ", self.before_proj(c).shape)
            print("x.shape: ", x.shape)
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c

class BaseCausalWanAttentionBlock(CausalWanAttentionBlock):
    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 block_id=None, # NEW
                 ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps) # pass through
        self.block_id = block_id

    def forward(
        self,
        x,
        hints, # NEW
        context_scale, # NEW
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        current_end=0
    ):
        x = super().forward(x,
                            e,
                            seq_lens,
                            grid_sizes,
                            freqs,
                            context,
                            context_lens,
                            block_mask,
                            kv_cache,
                            crossattn_cache,
                            current_start,
                            current_end,
                            )
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
        return x

    
class VaceCausalWanModel(CausalWanModel):
    @register_to_config
    def __init__(self,
                vace_layers=None, # NEW
                vace_in_dim=96, # NEW
                model_type='t2v',
                patch_size=(1, 2, 2),
                text_len=512,
                in_dim=16,
                dim=2048,
                ffn_dim=8192,
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=16,
                num_layers=32,
                window_size=(-1, -1),
                qk_norm=True,
                cross_attn_norm=True,
                eps=1e-6):

        super().__init__(model_type,
                        patch_size,
                        text_len,
                        in_dim,
                        dim,
                        ffn_dim,
                        freq_dim,
                        text_dim,
                        out_dim,
                        num_heads,
                        num_layers,
                        window_size,
                        qk_norm,
                        cross_attn_norm,
                        eps,
                        )

        self.vace_layers = [i for i in range(0, self.num_layers, 2)] if vace_layers is None else vace_layers
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

        assert 0 in self.vace_layers
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # OVERWRITE blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            BaseCausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    window_size, qk_norm, cross_attn_norm, eps,
                                    block_id = self.vace_layers_mapping[i] if i in self.vace_layers else None) # NEW
            for i in range(self.num_layers)
        ])

        # vace blocks
        assert cross_attn_type == 't2v_cross_attn' # IT SEEMS LIKE VACE ONLY SUPPORTS T2V 
        self.vace_blocks = nn.ModuleList([
            VaceWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
                                     self.cross_attn_norm, self.eps, block_id=i)
            for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        print("in dim? ", self.vace_in_dim)
        print("patch size?", self.patch_size)
        print("patch size? ", self.patch_size)
        print(self.vace_patch_embedding.state_dict())

    # based on https://github.com/ali-vilab/VACE/blob/main/vace/models/wan/modules/model.py
    # needs: 1. self.vace_patch_embedding, 2. self.vace_blocks
    def _forward_vace(self, x, vace_context, seq_len, kwargs):
        for item in vace_context:
            print("item in vace_context: ", item.shape) # each is 96, 21, 60, 104
        print("x.shape: ", x.shape) # 1, 4680, 1536 = 1, 3 frames * 60 * 104, hidden dimension = 1536
        # embeddings
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        for u in c:
            print("u shape after vace patch embedding: ", u.shape) # each is 1, 1536, 21, 30, 52
        c = [u.flatten(2).transpose(1, 2) for u in c] # each is 1, 32760, 1536
        for u in c:
            print("u shape after reshaping: ", u.shape) # each is 1, 32760, 1536
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])
        print("c shape: ", c.shape) # 1, 32760, 1536

        # arguments
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)

        for block in self.vace_blocks:
            c = block(c, **new_kwargs)
        hints = torch.unbind(c)[:-1]
        return hints

    def _forward_inference(
        self,
        x,
        t,
        context,
        vace_context,
        vace_context_scale,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        current_end: int = 0,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        # DO VACE HERE
        hints = self._forward_vace(x, vace_context, seq_len, kwargs)
        # DO VACE HERE

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                assert False
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "current_end": current_end
                    }
                )
                x = block(x, hints=hints, context_scale=vace_context_scale, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)