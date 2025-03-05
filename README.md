## FLUX tiled upscaler
A minimal tiled-image upscaler that uses [FLUX1.0-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and [ControlNet-Union](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro). This upscaler uses both tile and low quality controlnets to upscale images.

## Example Results
| Original | Upscaled |
|----------|----------|
| ![Original Portrait](https://github.com/neuralwork/flux-tiled-upscaler/blob/main/examples/portrait.jpeg) | ![Upscaled Portrait](https://github.com/neuralwork/flux-tiled-upscaler/blob/main/examples/outputs/output_portrait.png) |

## Installation
Tested with python 3.10, torch 2.1 and CUDA 11.8.

```sh
cd flux-tiled-upscaler
pip install -r requirements.txt

export HUGGINFACE_TOKEN="YOUR_TOKEN"
python scripts/download_models.py
```

You can then run the upscaler as follows:
```sh
# Upscale a single image by 2x
python run_upscaler.py --input_path examples/camp.jpg --output_folder results --upscale_factor 2

# Upscale a folder of images by 4x
python run_upscaler.py --input_path examples/ --output_folder results --upscale_factor 4

# Adjust quality settings
python run_upscaler.py --input_path examples/portrait.jpeg --num_inference_steps 12 --tile_control_scale 0.45 --low_quality_control_scale 0.7
```

Alternatively, you can launch the Gradio demo to use it interactively as follows:
```sh
# Run the interactive Gradio web demo
python gradio_demo.py
```

## License
This codebase is licensed under the [MIT license](https://github.com/neuralwork/flux-tiled-upscaler/blob/main/LICENSE). Please refer to the model pages of FLUX and ControlNet-Union for their licenses.

From [neuralwork](https://neuralwork.ai/) with :heart: