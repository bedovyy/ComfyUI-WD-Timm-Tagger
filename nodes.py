import csv
import os
from typing import NamedTuple

import numpy as np
import timm
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

import folder_paths
from comfy import model_management
from comfy.model_patcher import ModelPatcher

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("timm").setLevel(logging.WARNING)

# ------------------------------------------------------------------
# Reference: Original code from https://github.com/neggles/wdv3-timm
# ------------------------------------------------------------------
class LabelData(NamedTuple):
    names: list[str]
    rating: list[int]
    general: list[int]
    character: list[int]

def load_labels(path) -> LabelData:
    with open(os.path.join(path, "selected_tags.csv")) as f:
        rows = list(csv.DictReader(f))
    names = [r["name"] for r in rows]
    categories  = [int(r["category"]) for r in rows]

    return LabelData(
        names=names,
        rating=[i for i, c in enumerate(categories) if c == 9],
        general=[i for i, c in enumerate(categories) if c == 0],
        character=[i for i, c in enumerate(categories) if c == 4],
    )


MODEL_REPOS = [
    "SmilingWolf/wd-eva02-large-tagger-v3",
    "SmilingWolf/wd-vit-large-tagger-v3",
    "SmilingWolf/wd-vit-tagger-v3",
    "SmilingWolf/wd-swinv2-tagger-v3",
    "SmilingWolf/wd-convnext-tagger-v3",
]
REPO_NAMES = {repo.split("/")[-1]: repo for repo in MODEL_REPOS}
WD_TAGGER_DIR = os.path.join(folder_paths.models_dir, "wd_taggers")

def get_model_list() -> list[str]:
    local = set(os.listdir(WD_TAGGER_DIR)) if os.path.isdir(WD_TAGGER_DIR) else set()
    return list(REPO_NAMES) + sorted(local - REPO_NAMES.keys())


class WDTimmTagger:
    def __init__(self):
        self.model_patcher = None
        self.labels = None
        self.config = None
        self.current_model_name = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (get_model_list(), {"default": get_model_list()[0]}),
                "general_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "character_threshold": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "add_rating": ("BOOLEAN", {"default": False}),
                "exclude_tags": ("STRING", {"default": ""}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 32}),
            }
        }
    OUTPUT_IS_LIST = (True,True)
    RETURN_TYPES = ("STRING", "DICT")
    RETURN_NAMES = ("STRING", "RAW")
    FUNCTION = "tag"
    CATEGORY = "image"

    def tag(self, image, model_name, general_threshold, character_threshold, add_rating, exclude_tags, batch_size):
        device = model_management.get_torch_device()
        
        if self.current_model_name != model_name:
            save_path = os.path.join(WD_TAGGER_DIR, model_name)
            if not os.path.exists(os.path.join(save_path, "config.json")):
                repo_id = REPO_NAMES[model_name]
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=save_path,
                    allow_patterns=["model.safetensors", "config.json", "selected_tags.csv"],
                )

            base_model = timm.create_model(f"local-dir:{save_path}").eval()
            state_dict = load_file(os.path.join(save_path, "model.safetensors"))
            base_model.load_state_dict(state_dict)
            self.model_patcher = ModelPatcher(
                base_model,
                load_device=device,
                offload_device=model_management.intermediate_device()
            )
            self.labels = load_labels(save_path)
            self.config = timm.data.resolve_data_config(base_model.pretrained_cfg, model=base_model)
            self.current_model_name = model_name

        # resize then padding
        img_tensor = image.permute(0, 3, 1, 2)  # [B, C, H, W]
        B, C, H, W = img_tensor.shape
        _, target_h, target_w = self.config["input_size"]
        scale = min(target_w / W, target_h / H)
        new_w, new_h = int(W * scale), int(H * scale)

        resized = F.interpolate(img_tensor, size=(new_h, new_w), mode=self.config["interpolation"], align_corners=False)
        inputs = torch.ones((B, C, target_h, target_w))
        pad_y, pad_x = (target_h - new_h) // 2, (target_w - new_w) // 2
        inputs[:, :, pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        del resized

        mean = torch.tensor(self.config["mean"]).view(1, 3, 1, 1)
        std  = torch.tensor(self.config["std"]).view(1, 3, 1, 1)
        inputs = (inputs - mean) / std
        inputs = inputs[:, [2, 1, 0]]  # RGB → BGR

        required_mem = batch_size * int(np.prod(self.config["input_size"])) * 4 * 8
        model_management.free_memory(required_mem, device, keep_loaded=[self.model_patcher])
        model_management.load_model_gpu(self.model_patcher)

        def process_category(probs, indices, threshold):
            subset = {self.labels.names[i]: probs[i].item() for i in indices
                  if probs[i] > threshold and self.labels.names[i]}
            return dict(sorted(subset.items(), key=lambda x: x[1], reverse=True))

        def normalize(name: str) -> str:
            return name.replace("_", " ").replace("(", "\\(").replace(")", "\\)")

        exclude = {s.strip() for s in exclude_tags.lower().split(",") if s.strip()}
        results, raws = [], []
        for batch in torch.split(inputs, batch_size):
            with torch.inference_mode():
                batch_probs = F.sigmoid(self.model_patcher.model(batch.to(device))).cpu()

            for probs in batch_probs:
                probs_np = probs.numpy()
                ratings = {self.labels.names[i]: probs_np[i].item() for i in self.labels.rating}
                character = process_category(probs_np, self.labels.character, character_threshold)
                general = process_category(probs_np, self.labels.general, general_threshold)
                top_rating = [max(ratings, key=ratings.get)] if add_rating else []
                combined_tags = top_rating + list(character.keys()) + list(general.keys())
                taglist = ", ".join(normalized for t in combined_tags if (normalized := normalize(t)) not in exclude)
                results.append(taglist + (", " if taglist else ""))
                raws.append({"ratings": ratings, "character": character, "general": general})

        return (results, raws)
