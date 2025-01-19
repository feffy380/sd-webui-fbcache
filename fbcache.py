from typing import Optional

import torch as th
from sgm.modules.diffusionmodules.openaimodel import timestep_embedding


def are_two_tensors_similar(t1, t2, *, threshold):
    if t1.shape != t2.shape:
        return False
    mean_diff = (t1 - t2).abs().mean()
    mean_t1 = t1.abs().mean()
    diff = mean_diff / mean_t1
    return diff.item() < threshold


def call_remaining_blocks(self, hs, h, emb, context):
    """
    Call remaining blocks on cache miss
    """
    original_h = h

    for id, module in enumerate(self.input_blocks):
        if id < 2:
            continue
        h = module(h, emb, context)
        hs.append(h)

    h = self.middle_block(h, emb, context)
    for module in self.output_blocks:
        h = th.cat([h, hs.pop()], dim=1)
        h = module(h, emb, context)
    hidden_states_residual = h - original_h
    return h, hidden_states_residual


class FBCacheSession:
    """
    Session for First Block Cache, which holds cache data and provides functions for hooking the model.
    """
    def __init__(self, initial_step: int = 1):
        self.buffers: dict[str, th.Tensor] = {}
        self.stored_forward = None
        self.unet_reference = None
        self.current_sampling_step = initial_step
        self.consecutive_cache_hits = 0

    def get_buffer(self, name: str) -> Optional[th.Tensor]:
        return self.buffers.get(name)

    def set_buffer(self, name: str, buffer: th.Tensor):
        self.buffers[name] = buffer

    def increment_sampling_step(self):
        self.current_sampling_step += 1

    def apply_prev_hidden_states_residual(self, h: th.Tensor) -> th.Tensor:
        hidden_states_residual = self.get_buffer("hidden_states_residual")
        h = hidden_states_residual + h
        return h.contiguous()

    def get_can_use_cache(self, first_hidden_states_residual: th.Tensor, threshold: float) -> bool:
        prev_first_hidden_states_residual = self.get_buffer("first_hidden_states_residual")
        can_use_cache = prev_first_hidden_states_residual is not None and are_two_tensors_similar(
            prev_first_hidden_states_residual,
            first_hidden_states_residual,
            threshold=threshold,
        )
        return can_use_cache

    def hook_model(
            self,
            unet,
            steps: int,
            residual_diff_threshold: float,
            start: float = 0.0,
            end: float = 1.0,
            max_consecutive_cache_hits: int = -1
        ):
        cache = self

        using_validation = max_consecutive_cache_hits > 0 or start > 0.0 or end < 1.0
        validate_use_cache = None
        if using_validation:
            def validate_use_cache(use_cached):
                progress = cache.current_sampling_step / steps
                use_cached = use_cached and start < progress <= end
                use_cached = use_cached and (
                    max_consecutive_cache_hits < 0
                    or cache.consecutive_cache_hits < max_consecutive_cache_hits
                )
                cache.consecutive_cache_hits = cache.consecutive_cache_hits + 1 if use_cached else 0
                return use_cached

        def hijacked_unet_forward(
            self,
            x: th.Tensor,
            timesteps: Optional[th.Tensor] = None,
            context: Optional[th.Tensor] = None,
            y: Optional[th.Tensor] = None,
            **kwargs,
        ) -> th.Tensor:
            """
            Apply the model to an input batch.
            :param x: an [N x C x ...] Tensor of inputs.
            :param timesteps: a 1-D batch of timesteps.
            :param context: conditioning plugged in via crossattn
            :param y: an [N] Tensor of labels, if class-conditional.
            :return: an [N x C x ...] Tensor of outputs.
            """
            assert (y is not None) == (
                self.num_classes is not None
            ), "must specify y if and only if the model is class-conditional"
            hs = []
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)

            if self.num_classes is not None:
                assert y.shape[0] == x.shape[0]
                emb = emb + self.label_emb(y)

            can_use_cache = False

            h = x
            for id, module in enumerate(self.input_blocks):
                if id >= 2:
                    break
                if id == 1:
                    original_h = h
                h = module(h, emb, context)
                hs.append(h)
                if id == 1:
                    first_hidden_states_residual = h - original_h
                    can_use_cache = cache.get_can_use_cache(first_hidden_states_residual, threshold=residual_diff_threshold)
                    if validate_use_cache is not None:
                        can_use_cache = validate_use_cache(can_use_cache)
                    if not can_use_cache:
                        cache.set_buffer("first_hidden_states_residual", first_hidden_states_residual)
                    del first_hidden_states_residual

            if can_use_cache:
                h = cache.apply_prev_hidden_states_residual(h)
            else:
                h, hidden_states_residual = call_remaining_blocks(self, hs, h, emb, context)
                cache.set_buffer("hidden_states_residual", hidden_states_residual)

            h = h.type(x.dtype)

            return self.out(h)

        self.stored_forward = unet.forward
        unet.forward = hijacked_unet_forward.__get__(unet)
        unet._fbcache_hooked = True
        self.unet_reference = unet

    def detach(self):
        if self.unet_reference is None:
            return
        if not getattr(self.unet_reference, "_fbcache_hooked", False):
            return
        self.unet_reference.forward = self.stored_forward
        self.unet_reference._fbcache_hooked = False
        self.unet_reference = None
        self.stored_forward = None
