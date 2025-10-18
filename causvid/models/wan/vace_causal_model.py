from diffusers.configuration_utils import ConfigMixin, register_to_config
from causvid.models.wan.causal_model import CausalWanAttentionBlock, CausalWanModel
import torch.nn as nn

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
                vace_layers=None,
                vace_in_dim=None,
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

        # # vace blocks
        # self.vace_blocks = nn.ModuleList([
        #     VaceWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, self.num_heads, self.window_size, self.qk_norm,
        #                              self.cross_attn_norm, self.eps, block_id=i)
        #     for i in self.vace_layers
        # ])

        # # vace patch embeddings
        # self.vace_patch_embedding = nn.Conv3d(
        #     self.vace_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        # )
        