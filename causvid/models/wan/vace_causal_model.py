from causal_model import CausalWanAttentionBlock, CausalWanModel


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
                 block_id=None):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps) # pass through
        self.block_id = block_id

    # two new arguments: hints, context scale
    def forward(self, x, hints, seq_lens, grid_sizes, freqs, block_mask, kv_cache=None, current_start=0, current_end=0, context_scale=1.0,):
        x = super().forward(x, seq_lens, grid_sizes, freqs, block_mask, kv_cache, current_start, current_end) # direct passthrough
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
        return x

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

        