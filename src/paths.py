from pathlib import Path

# Directory base: dove si trova questo script
BASE_DIR = Path(__file__).resolve().parent

# Sottocartelle relative al progetto
MODELS_DIR = BASE_DIR / "model"
SR_SCRIPT_DIR = MODELS_DIR / 'SR_Script'
SR_SCRIPT_MODEL_DIR = SR_SCRIPT_DIR / "super_res"
IMAGES_DIR = BASE_DIR / "images"
IMAGES_OUTPUT_DIR = IMAGES_DIR / "output"
IMAGES_INPUT_DIR = IMAGES_DIR / "input"

IMAGES_DOWNSIZED_X2_DIR = IMAGES_OUTPUT_DIR / "personalized_downsizing_x2"
IMAGES_DOWNSIZED_X3_DIR = IMAGES_OUTPUT_DIR / "personalized_downsizing_x3"
IMAGES_DOWNSIZED_X4_DIR = IMAGES_OUTPUT_DIR / "personalized_downsizing_x4"

IMAGES_SR_X2_DIR = IMAGES_OUTPUT_DIR / "super_resolution_x2"
IMAGES_SR_X3_DIR = IMAGES_OUTPUT_DIR / "super_resolution_x3"
IMAGES_SR_X4_DIR = IMAGES_OUTPUT_DIR / "super_resolution_x4"
