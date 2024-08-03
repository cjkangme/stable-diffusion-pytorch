import torch
import numpy as np
from tqdm import tqdm
from models.ddpm import DDPMSampler

WIDTH, HEIGHT = 512, 512
LATENTS_WIDTH, LATENTS_HEIGHT = WIDTH // 8, HEIGHT // 8


def generate(
    prompt: str,
    uncond_prompt: str = "",
    input_image=None,
    strength=0.8,
    cfg_scale: float = 7.5,
    sampler_name: str = "ddpm",
    n_inference_steps: int = 50,
    batch_size: int = 1,
    models: dict = {},
    seed: int = -1,
    device: str = "cpu",
    idle_device=None,
    tokenizer=None,
):
    do_cfg = cfg_scale > 1

    with torch.no_grad():
        assert 0 < strength <= 1, "strength must be between 0 and 1"

        def to_idle(x):
            if idle_device:
                return x.to(idle_device)
            return x

        generator = torch.Generator(device=device)

        if seed == -1:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        # 프롬프트 임베딩
        if do_cfg:
            # (B, seq_len)
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            )
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            )
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)

            # (B, seq_len) -> (B, seq_len, dim)
            cond_embed = clip(cond_tokens)
            uncond_embed = clip(uncond_tokens)

            # (2B, 77, 768)
            context = torch.cat([cond_embed, uncond_embed], dim=0)
        else:
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            )
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (B, 77, 768)
            context = clip(cond_tokens)

        to_idle(clip)

        # Scheduler 초기화
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError("Not implemented")

        latent_shape = (batch_size, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image is not None:
            # Image to Image
            encoder = models["encoder"]
            encoder.to(device)

            # (H, W, C)
            input_image_tensor = input_image.resize(WIDTH, HEIGHT)
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            # TODO: Normalize 함수 구현 필요
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (H, W, C) -> (B, C, H, W)
            input_image_tensor = input_image_tensor.unsqueeze(0).permute(0, 3, 1, 2)

            # 노이즈 생성
            noise = torch.randn(latent_shape, generator=generator, device=device)
            # latents 인코딩 (VAE이므로 noise가 같이 들어감)
            latents = encoder(input_image_tensor, noise)

            sampler.set_strength(strength=strength)
            # noisy latents 획득
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # Text to Image
            latents = torch.randn(latent_shape, generator=generator, device=device)

        # UNet load
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timestpes)
        for i, timestep in enumerate(timesteps):
            # TODO: 타임 인코딩 함수 구현 필요
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (B, 4, latents_height, latents_weight)
            model_input = latents
            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)

            # Predict noise
            pred_noise = diffusion(model_input, context, time_embedding)
            if do_cfg:
                pred_cond, pred_uncond = pred_noise.chunk(2)
                pred_noise = cfg_scale * (pred_cond - pred_uncond) + pred_uncond

            # Remove Noise
            latents = sampler.step(timestep, latents, pred_noise)
        to_idle(diffusion)

        # 결과 확인하기
        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        # 이미지 후처리
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        # (H, W, C)
        return images[0]


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range

    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min

    if clamp:
        x = x.clamp(new_min, new_max)

    return x


def get_time_embedding(timestep):
    # (160, )
    freqs = torch.pow(10000, -torch.arange(0, 160, dtype=torch.float32) / 160)
    # (1, 1) * (1, 160) -> (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
