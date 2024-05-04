from base.pipeline import Pipeline
from transformers import CLIPTokenizer
from PIL import Image
from config.clip import CLIP
from config.encoder import VAE_Encoder
from config.decoder import VAE_Decoder
from config.diffusion import Diffusion
from base.load import get_outfit

import app.utils.model_converter as model_converter
import os

import torch


class Outfits:
    def __init__(self, prompt, negative_prompt=None):
        self.prompt = prompt
        self.negative_prompt = negative_prompt

    def getOutfit(self):
        images = []

        self.initializeParameters()

        pipeline = Pipeline()
        output_image = pipeline.generate(
            prompt=self.prompt,
            uncond_prompt=self.negative_prompt,
            input_image=self.input_image,
            strength=self.strength,
            do_cfg=self.do_cfg,
            cfg_scale=self.cfg_scale,
            sampler_name=self.sampler,
            n_inference_steps=self.num_inference_steps,
            models=self.models,
            seed=self.seed,
            device=self.DEVICE,
            idle_device="cpu",
            tokenizer=self.tokenizer,
        )

        img = Image.fromarray(output_image)
        img.save(
            "/Users/tanishqkakkar/MNIT/Sem-8/Major Project/Outfit_Designer/app/assets/output/outfit.jpg"
        )
        images.append("outfit.jpg")

        return images

    def initializeParameters(self):
        self.DEVICE = "cpu"
        self.ALLOW_CUDA = False
        self.ALLOW_MPS = False

        # Check for cuda or cpu
        if torch.cuda.is_available() and self.ALLOW_CUDA:
            self.DEVICE = "cuda"
        elif (torch.has_mps or torch.backends.mps.is_available()) and self.ALLOW_MPS:
            self.DEVICE = "mps"

        # Load fine tuned model
        self.tokenizer = CLIPTokenizer(
            vocab_file="/Users/tanishqkakkar/MNIT/Sem-8/Major Project/Outfit_Designer/app/assets/data/vocab.json",
            merges_file="/Users/tanishqkakkar/MNIT/Sem-8/Major Project/Outfit_Designer/app/assets/data/merges.txt",
        )
        self.model_file = "/Users/tanishqkakkar/MNIT/Sem-8/Major Project/Outfit_Designer/app/assets/data/model1.ckpt"
        self.models = self.preload_models_from_standard_weights(
            self.model_file, self.DEVICE
        )

        # Get outfit from inventory
        self.input_image = get_outfit(self.prompt)
        self.strength = 0.9

        # Give weight to unconditional/negative prompt
        self.do_cfg = True
        self.cfg_scale = 8

        # Initialize Sampler
        self.sampler = "ddpm"
        self.num_inference_steps = 50
        self.seed = 50

    def preload_models_from_standard_weights(self, ckpt_path, device):
        state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

        encoder = VAE_Encoder().to(device)
        encoder.load_state_dict(state_dict["encoder"], strict=True)

        decoder = VAE_Decoder().to(device)
        decoder.load_state_dict(state_dict["decoder"], strict=True)

        diffusion = Diffusion().to(device)
        diffusion.load_state_dict(state_dict["diffusion"], strict=True)

        clip = CLIP().to(device)
        clip.load_state_dict(state_dict["clip"], strict=True)

        return {
            "clip": clip,
            "encoder": encoder,
            "decoder": decoder,
            "diffusion": diffusion,
        }
