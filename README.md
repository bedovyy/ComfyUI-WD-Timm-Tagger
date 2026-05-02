# ComfyUI-WD-Timm-Tagger

ComfyUI custom node for image tagging using WD v3 models, based on `timm`.

## Installation

1. Clone this repo into `ComfyUI/custom_nodes/`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Performance

Tested on 291 images with RTX 5090, `wd-eva02-large-tagger-v3` model.

| Node | Time |
|---|---|
| [ComfyUI-WD14-Tagger](https://github.com/pythongosssss/ComfyUI-WD14-Tagger) | 226.99s |
| **ComfyUI-WD-Timm-Tagger** | **9.34s** |


## Models

Models are automatically downloaded to `models/wd_taggers/` on first use.

| Model | Repo |
|-------|------|
| eva02 | [SmilingWolf/wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3) |
| vit-large | [SmilingWolf/wd-vit-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-vit-large-tagger-v3) |
| vit | [SmilingWolf/wd-vit-tagger-v3](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3) |
| swinv2 | [SmilingWolf/wd-swinv2-tagger-v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3) |
| convnext | [SmilingWolf/wd-convnext-tagger-v3](https://huggingface.co/SmilingWolf/wd-convnext-tagger-v3) |

You can also place your own model folders under `models/wd_taggers/` and they will appear in the model list automatically.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Input image(s) |
| `model_name` | COMBO | Model to use |
| `general_threshold` | FLOAT | Confidence threshold for general tags (default: 0.35) |
| `character_threshold` | FLOAT | Confidence threshold for character tags (default: 0.75) |
| `add_rating` | BOOLEAN | Prepend rating tag to output |
| `exclude_tags` | STRING | Comma-separated tags to exclude |
| `batch_size` | INT | Number of images per inference batch |

> `batch_size` only takes effect when multiple images are passed as a batch.

## Outputs

- **STRING** — Formatted tag string per image
- **RAW** — Dict with `ratings`, `character`, `general` scores per image

## Credits

Based on [wdv3-timm](https://github.com/neggles/wdv3-timm) by neggles.
