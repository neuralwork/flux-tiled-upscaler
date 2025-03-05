import os
import random

import torch
import spaces
import gradio as gr
import numpy as np
from huggingface_hub import snapshot_download
from run_upscaler import ImageUpscaler

css = """
#col-container {
    margin: 0 auto;
    max-width: 512px;
}
"""

if torch.cuda.is_available():
    power_device = "GPU"
    device = "cuda"
else:
    power_device = "CPU"
    device = "cpu"

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev", 
    repo_type="model", 
    ignore_patterns=["*.md", "*..gitattributes"],
    local_dir="FLUX.1-dev",
    token=huggingface_token,
)

model_path = snapshot_download(
    repo_id="Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro", 
    repo_type="model", 
    ignore_patterns=["*.md", "*..gitattributes"],
    local_dir="controlnet-union",
    token=huggingface_token,
)

MAX_SEED = 1000000


@spaces.GPU
def infer(
    seed,
    randomize_seed,
    input_image,
    num_inference_steps,
    upscale_factor,
    tile_control_scale,
    low_quality_control_scale,
    guidance_scale,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # Initialize the upscaler with our pipeline components
    upscaler = ImageUpscaler(
        upscale_factor=upscale_factor,
        tile_control_scale=tile_control_scale,
        low_quality_control_scale=low_quality_control_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    )

    # Upscale the image - our fixed upscaler will handle all sizing correctly
    final_image = upscaler.upscale_image(input_image)

    return final_image

def create_snow_effect():
    snow_css = """
    @keyframes snowfall {
        0% {
            transform: translateY(-10vh) translateX(0);
            opacity: 1;
        }
        100% {
            transform: translateY(100vh) translateX(100px);
            opacity: 0.3;
        }
    }
    .snowflake {
        position: fixed;
        color: white;
        font-size: 1.5em;
        user-select: none;
        z-index: 1000;
        pointer-events: none;
        animation: snowfall linear infinite;
    }
    """

    snow_js = """
    function createSnowflake() {
        const snowflake = document.createElement('div');
        snowflake.innerHTML = '❄';
        snowflake.className = 'snowflake';
        snowflake.style.left = Math.random() * 100 + 'vw';
        snowflake.style.animationDuration = Math.random() * 3 + 2 + 's';
        snowflake.style.opacity = Math.random();
        document.body.appendChild(snowflake);
        
        setTimeout(() => {
            snowflake.remove();
        }, 5000);
    }
    setInterval(createSnowflake, 200);
    """

    # CSS and JavaScript combined HTML
    snow_html = f"""
    <style>
        {snow_css}
    </style>
    <script>
        {snow_js}
    </script>
    """
    
    return gr.HTML(snow_html)

with gr.Blocks(theme="Yntec/HaleyCH_Theme_Orange", css=css) as demo:

    create_snow_effect()    

    with gr.Row():
        with gr.Column(scale=1):
            input_im = gr.Image(label="Input Image", type="pil")
        with gr.Column(scale=1):
            result = gr.Image(label="Output Image", type="pil")

    with gr.Row():
        num_inference_steps = gr.Slider(
            label="Number of Inference Steps",
            minimum=8,
            maximum=50,
            step=1,
            value=12,
        )
        upscale_factor = gr.Slider(
            label="Upscale Factor",
            minimum=1,
            maximum=4,
            step=1,
            value=2,
        )
        tile_control_scale = gr.Slider(
            label="Tile ControlNet Conditioning Scale",
            minimum=0.1,
            maximum=1.0,
            step=0.1,
            value=0.45,
        )
        low_quality_control_scale = gr.Slider(
            label="Low Quality ControlNet Conditioning Scale",
            minimum=0.1,
            maximum=1.0,
            step=0.1,
            value=0.2,
        )
        guidance_scale = gr.Slider(
            label="Guidance Scale",
            minimum=1.0,
            maximum=10.0,
            step=0.1,
            value=3.5,
        )
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=42,
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

    with gr.Row():
        run_button = gr.Button(value="Run")

    examples = gr.Examples(
        examples=[
            [42, False, "examples/anime1.png", 12, 2, 0.45, 0.2, 3.5],
            [42, False, "examples/portrait.jpeg", 12, 2, 0.45, 0.7, 3.5],
        ],
        inputs=[
            seed,
            randomize_seed,
            input_im,
            num_inference_steps,
            upscale_factor,
            tile_control_scale,
            low_quality_control_scale,
            guidance_scale,
        ],
        fn=infer,
        outputs=result,
        cache_examples="lazy",
    )

    run_button.click(
        fn=infer,
        inputs=[
            seed,
            randomize_seed,
            input_im,
            num_inference_steps,
            upscale_factor,
            tile_control_scale,
            low_quality_control_scale,
            guidance_scale,
        ],
        outputs=result,
    )

demo.queue().launch(share=False, show_api=False)