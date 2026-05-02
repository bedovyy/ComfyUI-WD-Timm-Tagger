from .nodes import WDTimmTagger
import folder_paths
import os

folder_paths.add_model_folder_path(
    "wd_taggers",
    os.path.join(folder_paths.models_dir, "wd_taggers")
)

NODE_CLASS_MAPPINGS = {
    "WDTimmTagger": WDTimmTagger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WDTimmTagger": "WD Timm Tagger",
}
