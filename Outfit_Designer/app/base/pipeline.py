import torch
from tqdm import tqdm
from config.ddpm import DDPMSampler


class Pipeline:
    def __init__(self):
        self.WIDTH = 512
        self.HEIGHT = 512
        self.LATENTS_WIDTH = self.WIDTH // 8
        self.LATENTS_HEIGHT = self.HEIGHT // 8

    def generate(
        self,
        prompt,
        uncond_prompt=None,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name="ddpm",
        n_inference_steps=50,
        models={},
        seed=None,
        device=None,
        idle_device=None,
        tokenizer=None,
    ):
        with torch.no_grad():
            if idle_device:
                to_idle = lambda x: x.to(idle_device)
            else:
                to_idle = lambda x: x

            # Initialize random number generator according to the seed specified
            generator = torch.Generator(device=device)
            if seed is None:
                generator.seed()
            else:
                generator.manual_seed(seed)

            clip = models["clip"]
            clip.to(device)

            if do_cfg:
                # Convert into a list of length Seq_Len=77
                cond_tokens = tokenizer.batch_encode_plus(
                    [prompt], padding="max_length", max_length=77
                ).input_ids
                # (Batch_Size, Seq_Len)
                cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
                # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
                cond_context = clip(cond_tokens)
                # Convert into a list of length Seq_Len=77
                uncond_tokens = tokenizer.batch_encode_plus(
                    [uncond_prompt], padding="max_length", max_length=77
                ).input_ids
                # (Batch_Size, Seq_Len)
                uncond_tokens = torch.tensor(
                    uncond_tokens, dtype=torch.long, device=device
                )
                # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
                uncond_context = clip(uncond_tokens)
                # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
                context = torch.cat([cond_context, uncond_context])
            else:
                # Convert into a list of length Seq_Len=77
                tokens = tokenizer.batch_encode_plus(
                    [prompt], padding="max_length", max_length=77
                ).input_ids
                # (Batch_Size, Seq_Len)
                tokens = torch.tensor(tokens, dtype=torch.long, device=device)
                # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
                context = clip(tokens)
            to_idle(clip)

            if sampler_name == "ddpm":
                sampler = DDPMSampler(generator)
                sampler.set_inference_timesteps(n_inference_steps)
            else:
                raise ValueError("Unknown sampler value %s. ")

            latents_shape = (1, 4, self.LATENTS_HEIGHT, self.LATENTS_WIDTH)
            latents = torch.randn(latents_shape, generator=generator, device=device)

            diffusion = models["diffusion"]
            diffusion.to(device)

            timesteps = tqdm(sampler.timesteps)
            for i, timestep in enumerate(timesteps):
                # (1, 320)
                time_embedding = self.get_time_embedding(timestep).to(device)

                # (Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = latents

                if do_cfg:
                    # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                    model_input = model_input.repeat(2, 1, 1, 1)

                # model_output is the predicted noise
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
                model_output = diffusion(model_input, context, time_embedding)

                if do_cfg:
                    output_cond, output_uncond = model_output.chunk(2)
                    model_output = (
                        cfg_scale * (output_cond - output_uncond) + output_uncond
                    )

                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
                latents = sampler.step(timestep, latents, model_output)

            to_idle(diffusion)

            decoder = models["decoder"]
            decoder.to(device)
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
            images = decoder(latents)
            to_idle(decoder)

            images = self.rescale(images, (-1, 1), (0, 255), clamp=True)
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
            images = images.permute(0, 2, 3, 1)
            images = images.to("cpu", torch.uint8).numpy()
            return images[0]

    def rescale(self, x, old_range, new_range, clamp=False):
        old_min, old_max = old_range
        new_min, new_max = new_range
        x -= old_min
        x *= (new_max - new_min) / (old_max - old_min)
        x += new_min
        if clamp:
            x = x.clamp(new_min, new_max)
        return x

    def get_time_embedding(self, timestep):
        # Shape: (160,)
        freqs = torch.pow(
            10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160
        )
        # Shape: (1, 160)
        x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
        # Shape: (1, 160 * 2)
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)