import os
import math
import argparse

import torch
import numpy as np
from PIL import Image
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline, FluxControlNetModel, FluxMultiControlNetModel
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Image Upscaling with ControlNet")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input image or folder")
    parser.add_argument("--output_folder", type=str, default="./", help="Folder to save the output images (default: ./)")
    parser.add_argument("--upscale_factor", type=int, choices=[2, 4], default=2, help="Upscale factor (default: 2)")
    parser.add_argument("--tile_control_scale", type=float, default=0.45, help="ControlNet conditioning scale for tile (default: 0.45)")
    parser.add_argument("--low_quality_control_scale", type=float, default=0.2, help="ControlNet conditioning scale for low quality (default: 0.2)")
    parser.add_argument("--num_inference_steps", type=int, default=12, help="Number of inference steps (default: 12)")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale (default: 3.5)")
    return parser.parse_args()


class ImageUpscaler:
    def __init__(self, upscale_factor, tile_control_scale, low_quality_control_scale, num_inference_steps, guidance_scale):
        self.upscale_factor = upscale_factor
        self.tile_control_scale = tile_control_scale
        self.low_quality_control_scale = low_quality_control_scale
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.pipe = self.initialize_pipeline()

    def initialize_pipeline(self):
        # load pre-computed empty text prompt embeddings
        null_text_embeds = torch.load('null_text_embeds.pt')
        self.prompt_embeds = null_text_embeds['prompt_embeds']
        self.pooled_prompt_embeds = null_text_embeds['pooled_prompt_embeds']

        # load model
        controlnet_union = FluxControlNetModel.from_pretrained(
            "./controlnet-union", 
            torch_dtype=torch.bfloat16
        )
        controlnet = FluxMultiControlNetModel([controlnet_union]) 

        pipe = FluxControlNetPipeline.from_pretrained(
            "./FLUX.1-dev", 
            controlnet=controlnet, 
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            torch_dtype=torch.bfloat16
        )
        pipe.to("cuda")
        return pipe

    def upscale_image(self, input_image):
        original_width, original_height = input_image.size
        
        target_width = math.ceil(original_width * self.upscale_factor)
        target_height = math.ceil(original_height * self.upscale_factor)
        
        # we need images to be at least 1024x1024 
        # check if we need to use a larger intermediate size
        if min(target_width, target_height) < 1024:
            # compute how much we need to scale up to reach minimum 1024 in smallest dimension
            process_scale = 1024 / min(original_width, original_height)
            process_width = math.ceil(original_width * process_scale)
            process_height = math.ceil(original_height * process_scale)
            
            # interpolate the image to processing size
            process_image = input_image.resize((process_width, process_height), resample=Image.LANCZOS)
            grid = split_grid(process_image, tile_width=1024, tile_height=1024, overlap=128)
        else:
            process_image = input_image.resize((target_width, target_height), resample=Image.LANCZOS)
            grid = split_grid(process_image, tile_width=1024, tile_height=1024, overlap=128)

        # queue tiles for inference
        work = []
        for _y, _h, row in grid.tiles:
            for tiledata in row:
                work.append(tiledata[2])

        print(f"Number of tiles: {len(work)}")
        batch_size = min(len(work), 1)  # batch size has no impact on inference speed for flux
        batch_count = math.ceil(len(work) / batch_size)
        work_results = []

        for i in range(batch_count):
            tile_image = work[i * batch_size:(i + 1) * batch_size]

            outputs = self.pipe(
                prompt_embeds=torch.repeat_interleave(self.prompt_embeds, batch_size, dim=0),
                pooled_prompt_embeds=torch.repeat_interleave(self.pooled_prompt_embeds, batch_size, dim=0),
                control_image=[tile_image, tile_image],
                control_mode=[1, 6],  # tile, low quality
                width=1024,
                height=1024,
                controlnet_conditioning_scale=[self.tile_control_scale, self.low_quality_control_scale],
                num_inference_steps=self.num_inference_steps, 
                guidance_scale=self.guidance_scale,
                generator=torch.manual_seed(42),
            )
            work_results += outputs.images

        image_index = 0
        for _y, _h, row in grid.tiles:
            for tiledata in row:
                if image_index < len(work_results):
                    tiledata[2] = work_results[image_index]
                else:
                    tiledata[2] = Image.new("RGB", (1024, 1024))
                image_index += 1

        # stitch grid to create final image
        final_image = combine_grid(grid)

        # resize to the target dimensions (original × upscale_factor)
        final_image = final_image.resize((target_width, target_height), resample=Image.LANCZOS)

        return final_image

    def process(self, input_path, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if os.path.isdir(input_path):
            for filename in os.listdir(input_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    input_image = load_image(os.path.join(input_path, filename)).convert("RGB")
                    final_image = self.upscale_image(input_image)
                    final_image.save(os.path.join(output_folder, f"upscaled_{filename}"))
        else:
            input_image = load_image(input_path).convert("RGB")
            final_image = self.upscale_image(input_image)
            final_image.save(os.path.join(output_folder, "output.png"))


def main():
    args = parse_args()
    upscaler = ImageUpscaler(
        args.upscale_factor,
        args.tile_control_scale,
        args.low_quality_control_scale,
        args.num_inference_steps,
        args.guidance_scale
    )
    upscaler.process(args.input_path, args.output_folder)


if __name__ == "__main__":
    main()