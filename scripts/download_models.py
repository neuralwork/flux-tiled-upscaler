import os
from huggingface_hub import snapshot_download


huggingface_token = os.getenv("HUGGINFACE_TOKEN")

model_path = snapshot_download(
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
