import gradio as gr
from fbcache import FBCacheSession

from modules import processing, script_callbacks, scripts, shared
from modules.sd_samplers_common import setup_img2img_steps
from modules.ui_components import InputAccordion


class FBCacheScript(scripts.Script):
    def __init__(self):
        self.session: FBCacheSession = None
        script_callbacks.on_cfg_after_cfg(self.next_sampling_step)

    def title(self):
        return "First Block Cache"
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title()) as enabled:
            residual_diff_threshold = gr.Slider(
                label="Residual difference threshold",
                info="Cache matching tolerance - lower values are stricter. Set to 0 to disable FBCache.",
                value=0.2, minimum=0.0, maximum=1.0, step=0.001,
            )
            with gr.Row():
                start = gr.Slider(
                    label="Start",
                    info="When to enable caching (as % of sampling time). 0.0 = start, 0.5 = halfway point.",
                    value=0.0, minimum=0.0, maximum=1.0, step=0.01,
                )
                end = gr.Slider(
                    label="End",
                    info="When to disable caching (as % of sampling time). 1.0 = end, 0.5 = halfway point.",
                    value=1.0, minimum=0.0, maximum=1.0, step=0.01,
                )
            max_consecutive_cache_hits = gr.Number(
                label="Max consecutive cache hits",
                info="Limits consecutive cached results before full model call is forced. -1 = unlimited, 0 = disable FBCache.",
                value=-1, minimum=-1, step=1,
            )

        components = [enabled, residual_diff_threshold, start, end, max_consecutive_cache_hits]
        for component in components:
            component.do_not_save_to_config = True

        return components

    def process_batch(self, p: processing.StableDiffusionProcessing, enabled, residual_diff_threshold, start, end, max_consecutive_cache_hits, *args, **kwargs):
        if enabled and residual_diff_threshold > 0.0 and max_consecutive_cache_hits != 0:
            total_steps = p.steps
            initial_step = 1
            if self.is_img2img:
                total_steps, steps = setup_img2img_steps(p)
                initial_step = total_steps - steps
            self.configure_fbcache(total_steps, initial_step, residual_diff_threshold, start, end, max_consecutive_cache_hits)

    def before_hr(self, p: processing.StableDiffusionProcessing, enabled, residual_diff_threshold, start, end, max_consecutive_cache_hits, *args):
        self.detach_fbcache()
        if enabled and residual_diff_threshold > 0.0 and max_consecutive_cache_hits != 0:
            total_steps = getattr(p, "hr_second_pass_steps", 0) or p.steps
            total_steps, steps = setup_img2img_steps(p, total_steps)  # hires fix doesn't reduce steps based on denoise strength
            initial_step = total_steps - steps
            self.configure_fbcache(total_steps, initial_step, residual_diff_threshold, start, end, max_consecutive_cache_hits)

    def postprocess_batch(self, p: processing.StableDiffusionProcessing, *args, **kwargs):
        self.detach_fbcache()

    def next_sampling_step(self, *args):
        if self.session is not None:
            self.session.next_sampling_step()

    def configure_fbcache(self, steps, initial_step, residual_diff_threshold, start, end, max_consecutive_cache_hits):
        if self.session is None:
            self.session = FBCacheSession(initial_step)
        self.session.hook_model(shared.sd_model.model.diffusion_model, steps, residual_diff_threshold, start, end, max_consecutive_cache_hits)

    def detach_fbcache(self):
        if self.session is not None:
            self.session.detach()
            self.session = None
