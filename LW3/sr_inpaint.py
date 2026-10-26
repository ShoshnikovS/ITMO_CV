"""ЛР3.2. SR-inpainting на предобученной диффузионной модели Stable Diffusion x4 Upscaler.

Идея: обычный сэмплинг превращает шум в 4x-увеличенное изображение, условием служит LR-картинка.
Мы меняем процесс сэмплинга: на каждом шаге области ВНЕ маски заменяются на зашумлённую (до текущего
уровня шума) версию «неулучшенного» изображения (bicubic-увеличение входа). Так диффузия дорисовывает
детали только в области маски (mask=1), а остальное остаётся как есть. В конце результат ещё раз
смешивается с bicubic-изображением по маске в пикселях, чтобы вне маски совпадение было точным.
"""
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image

MODEL = "stabilityai/stable-diffusion-x4-upscaler"
SCALE = 4


def load_pipe():
    pipe = StableDiffusionUpscalePipeline.from_pretrained(MODEL, variant="fp16", torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def sr_inpaint(pipe, lr, mask, steps=20, noise_level=20, guidance=1.0, prompt="", seed=0):
    """lr: PIL RGB (w, h). mask: float-массив (4h, 4w) из 0/1 (1 - улучшать). Возвращает PIL RGB (4w, 4h)."""
    if mask.sum() == 0:  # маска из нулей: результат по построению равен bicubic, диффузию не запускаем
        return bicubic_up(lr)
    gen = torch.Generator().manual_seed(seed)
    x_lr = pipe.image_processor.preprocess(lr)  # (1,3,h,w) в [-1,1]
    h, w = x_lr.shape[2:]
    bicubic = F.interpolate(x_lr, scale_factor=SCALE, mode="bicubic", align_corners=False).clamp(-1, 1)
    m_pix = torch.from_numpy(mask.astype(np.float32))[None, None]
    m_lat = (F.interpolate(m_pix, size=(h, w), mode="area") > 0).float()  # маска на сетке латентов

    use_cfg = guidance > 1.0
    prompt_embeds, neg_embeds = pipe.encode_prompt(prompt, "cpu", 1, use_cfg, "" if use_cfg else None)
    if use_cfg:
        prompt_embeds = torch.cat([neg_embeds, prompt_embeds])

    # условие: зашумлённая LR-картинка (как в оригинальном пайплайне)
    nl = torch.tensor([noise_level], dtype=torch.long)
    cond = pipe.low_res_scheduler.add_noise(x_lr, torch.randn(x_lr.shape, generator=gen), nl)
    k = 2 if use_cfg else 1
    cond, nl_in = torch.cat([cond] * k), torch.cat([nl] * k)

    # «известное» изображение в латентном пространстве и фиксированный шум для его зашумления
    z_known = pipe.vae.encode(bicubic).latent_dist.mean * pipe.vae.config.scaling_factor
    known_noise = torch.randn(z_known.shape, generator=gen)

    pipe.scheduler.set_timesteps(steps)
    timesteps = pipe.scheduler.timesteps
    latents = torch.randn((1, pipe.vae.config.latent_channels, h, w), generator=gen) * pipe.scheduler.init_noise_sigma

    for i, t in enumerate(timesteps):
        inp = torch.cat([latents] * k)
        inp = torch.cat([pipe.scheduler.scale_model_input(inp, t), cond], dim=1)
        eps = pipe.unet(inp, t, encoder_hidden_states=prompt_embeds, class_labels=nl_in, return_dict=False)[0]
        if use_cfg:
            e_u, e_t = eps.chunk(2)
            eps = e_u + guidance * (e_t - e_u)
        latents = pipe.scheduler.step(eps, t, latents, return_dict=False)[0]
        if i + 1 < len(timesteps):  # известная часть, зашумлённая до уровня следующего шага
            known_t = pipe.scheduler.add_noise(z_known, known_noise, timesteps[i + 1:i + 2])
        else:
            known_t = z_known
        latents = m_lat * latents + (1 - m_lat) * known_t

    sr = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0].clamp(-1, 1)
    out = m_pix * sr + (1 - m_pix) * bicubic  # точное совпадение с bicubic вне маски
    arr = ((out[0].permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def bicubic_up(lr):
    """Baseline: обычное bicubic-увеличение (то же, что получается при mask = 0)."""
    return lr.resize((lr.width * SCALE, lr.height * SCALE), Image.BICUBIC)
