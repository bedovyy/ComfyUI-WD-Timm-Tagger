import csv
import fnmatch
import hashlib
import importlib
import json
import os
import sys
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
    rating: np.ndarray
    general: np.ndarray
    character: np.ndarray
    best_threshold: np.ndarray

def load_labels(path) -> LabelData:
    with open(os.path.join(path, "selected_tags.csv")) as f:
        rows = list(csv.DictReader(f))
    names = [r["name"] for r in rows]
    categories = np.array([int(r["category"]) for r in rows])
    best_threshold = np.array([float(r.get("best_threshold", 0.0)) for r in rows], dtype=np.float32)

    return LabelData(
        names=names,
        rating=np.where(categories == 9)[0],
        general=np.where(np.isin(categories, [0, 1, 5]))[0],
        character=np.where(np.isin(categories, [3, 4]))[0],
        best_threshold=best_threshold,
    )

DTYPE_MAP = {"auto": None, "bf16": torch.bfloat16, "fp16": torch.float16}
MODEL_REPOS = [
    "SmilingWolf/wd-eva02-large-tagger-v3",
    "SmilingWolf/wd-vit-large-tagger-v3",
    "SmilingWolf/wd-vit-tagger-v3",
    "SmilingWolf/wd-swinv2-tagger-v3",
    "SmilingWolf/wd-convnext-tagger-v3",
    "Bedovyy/pixai-tagger-v1.0-timm",
    "Bedovyy/pixai-tagger-v0.9-timm",
    "Bedovyy/OppaiOracle-V1.1-timm",
]

REPO_NAMES = {repo.split("/")[-1]: repo for repo in MODEL_REPOS}
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WD_TAGGER_DIR = os.path.join(folder_paths.models_dir, "wd_taggers")

def get_model_list() -> list[str]:
    local = set(os.listdir(WD_TAGGER_DIR)) if os.path.isdir(WD_TAGGER_DIR) else set()
    return list(REPO_NAMES) + sorted(local - REPO_NAMES.keys())

def parse_notes(pretrained_cfg: dict) -> dict:
    for note in pretrained_cfg.get('notes', []):
        try:
            return json.loads(note)
        except (json.JSONDecodeError, TypeError):
            continue
    return {}

ARCH_FILE = "timm_arch.py"
_loaded = set()

def load_custom_arch(save_path):
    py = os.path.realpath(os.path.join(save_path, ARCH_FILE))
    if not os.path.isfile(py) or py in _loaded:
        return
    name = "timm_arch_" + hashlib.md5(py.encode()).hexdigest()[:8]
    spec = importlib.util.spec_from_file_location(name, py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _loaded.add(py)

def cast_keep_complex(module: torch.nn.Module, dtype) -> torch.nn.Module:
    if dtype is None:
        return module
    for tensor in list(module.parameters()) + list(module.buffers()):
        if not torch.is_complex(tensor):
            tensor.data = tensor.data.to(dtype)
    return module

class WDTimmTagger:
    def __init__(self):
        self.model_patcher = None
        self.labels = None
        self.pad_color = None
        self.color_space = None
        self.antialias = False
        self.config = None
        self.current_model_name = None
        self.dtype = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (get_model_list(), {"default": get_model_list()[0]}),
                "dtype": (list(DTYPE_MAP.keys()), {"default": "auto"}),
                "general_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "character_threshold": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "add_rating": ("BOOLEAN", {"default": False}),
                "exclude_tags": ("STRING", {
                    "default": "",
                    "tooltip": "Exclude tags via comma separation. Glob patterns supported.",
                }),
                "batch_size": ("INT", {
                    "default": 4, "min": 1, "max": 32,
                    "tooltip": "Only effective when processing image batches.",
                }),
            }
        }
    OUTPUT_IS_LIST = (True,True)
    RETURN_TYPES = ("STRING", "DICT")
    RETURN_NAMES = ("STRING", "RAW")
    FUNCTION = "tag"
    CATEGORY = "image"

    def tag(self, image, model_name, dtype, general_threshold, character_threshold, add_rating, exclude_tags, batch_size):
        device = model_management.get_torch_device()
        
        if self.current_model_name != model_name or self.dtype != dtype:
            save_path = os.path.join(WD_TAGGER_DIR, model_name)
            if not os.path.exists(os.path.join(save_path, "config.json")):
                repo_id = REPO_NAMES[model_name]
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=save_path,
                    allow_patterns=["model.safetensors", "config.json", "selected_tags.csv", "preprocess.json", ARCH_FILE],
                )

            load_custom_arch(save_path)
            base_model = timm.create_model(f"local-dir:{save_path}", scriptable=False).eval()
            state_dict = load_file(os.path.join(save_path, "model.safetensors"))
            self.dtype = dtype
            dtype_to = DTYPE_MAP[dtype] or next(iter(state_dict.values())).dtype
            base_model.load_state_dict(state_dict)
            base_model = cast_keep_complex(base_model, dtype_to)

            self.model_patcher = ModelPatcher(
                base_model,
                load_device=device,
                offload_device=model_management.intermediate_device()
            )
            self.labels = load_labels(save_path)
            self.config = timm.data.resolve_data_config(base_model.pretrained_cfg, model=base_model)
            meta = parse_notes(base_model.pretrained_cfg)
            pad = meta.get("pad_color", 255)
            self.pad_color = None if pad is None else pad / 255.0
            self.antialias = bool(meta.get("antialias", False))
            self.color_space = meta.get("color_space") or (
                "RGB" if os.path.exists(os.path.join(save_path, "preprocess.json")) else "BGR"
            )
            self.current_model_name = model_name

        # resize then padding
        img_tensor = image.permute(0, 3, 1, 2)  # [B, C, H, W]
        B, C, H, W = img_tensor.shape
        _, target_h, target_w = self.config["input_size"]

        interp_kwargs = {"align_corners": False}
        if self.antialias:
            interp_kwargs["antialias"] = True

        if self.pad_color is None:
            inputs = F.interpolate(img_tensor, size=(target_h, target_w), mode="bicubic", **interp_kwargs)
        else:
            scale = min(target_w / W, target_h / H)
            new_w, new_h = int(W * scale), int(H * scale)

            resized = F.interpolate(
                img_tensor, size=(new_h, new_w), mode=self.config["interpolation"], **interp_kwargs
            )
            inputs = torch.full((B, C, target_h, target_w), self.pad_color)
            pad_y, pad_x = (target_h - new_h) // 2, (target_w - new_w) // 2
            inputs[:, :, pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
            del resized

        mean = torch.tensor(self.config["mean"]).view(1, 3, 1, 1)
        std  = torch.tensor(self.config["std"]).view(1, 3, 1, 1)
        inputs = (inputs - mean) / std
        if self.color_space == "BGR":
            inputs = inputs[:, [2, 1, 0]]  # RGB → BGR

        required_mem = batch_size * int(np.prod(self.config["input_size"])) * 4 * 8
        model_management.free_memory(required_mem, device, keep_loaded=[self.model_patcher])
        model_management.load_model_gpu(self.model_patcher)

        def process_category(probs, indices, threshold):
            effective_threshold = np.maximum(self.labels.best_threshold[indices], threshold)
            subset = {self.labels.names[i]: probs[i].item() for i, thr in zip(indices, effective_threshold)
                if probs[i] > thr and self.labels.names[i]}
            return dict(sorted(subset.items(), key=lambda x: x[1], reverse=True))

        def normalize(name: str) -> str:
            return name.replace("_", " ").replace("(", "\\(").replace(")", "\\)")

        exclude_patterns = [s.strip() for s in exclude_tags.lower().split(",") if s.strip()]
        results, raws = [], []
        model_dtype = next(self.model_patcher.model.parameters()).dtype
        ac_dtype = torch.float16 if model_dtype == torch.float16 else None
        for batch in torch.split(inputs, batch_size):
            with torch.inference_mode():
                with torch.autocast(device_type=device.type, dtype=ac_dtype, enabled=(ac_dtype is not None)):
                    logits = self.model_patcher.model(batch.to(device=device, dtype=model_dtype))
                assert torch.isfinite(logits).all(), (f"dtype {model_dtype} produced NaN/Inf. Please try other dtypes.")
                batch_probs = F.sigmoid(logits.to(torch.float32)).cpu()

            for probs in batch_probs:
                probs_np = probs.numpy()
                ratings = {self.labels.names[i]: probs_np[i].item() for i in self.labels.rating}
                character = process_category(probs_np, self.labels.character, character_threshold)
                general = process_category(probs_np, self.labels.general, general_threshold)
                top_rating = [max(ratings, key=ratings.get)] if add_rating and ratings else []
                combined_tags = top_rating + list(character.keys()) + list(general.keys())
                def is_excluded(n):
                    return any(fnmatch.fnmatchcase(n, p) for p in exclude_patterns)
                taglist = ", ".join(nt for t in combined_tags if (nt := normalize(t)) and not is_excluded(nt))
                results.append(taglist + (", " if taglist else ""))
                raws.append({"ratings": ratings, "character": character, "general": general})

        return (results, raws)
