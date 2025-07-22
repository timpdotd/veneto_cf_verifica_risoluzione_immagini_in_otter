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
IMAGES_DOWNSIZED_DIR = IMAGES_OUTPUT_DIR / "personalized_downsizing"
IMAGES_RESIZED_DIR = IMAGES_OUTPUT_DIR / "resized"

IMAGES_SR_X2_DIR = IMAGES_OUTPUT_DIR / "super_resolution_x2"
IMAGES_SR_X3_DIR = IMAGES_OUTPUT_DIR / "super_resolution_x3"
IMAGES_SR_X4_DIR = IMAGES_OUTPUT_DIR / "super_resolution_x4"

# Scala per la super risoluzione (cambiare se volete diversa da x2)
SUPER_RESOLUTION_PAR = 2

# Proporzioni (sono empiriche, cioè misurate dalle foto)
# NON TOCCARE
# Le misure sono in px o in mm
CHROMATIC_SCALE_RULER_MM = 200 #mm
INCH_CONVERSION = 25.4
CHROMATIC_SCALE_RULER_INCH = CHROMATIC_SCALE_RULER_MM/INCH_CONVERSION

# 400 PPI
TARGET_RULER_MM_OUTPUT = 200  # Desired ruler size in mm
TARGET_RULER_INCH_OUTPUT = TARGET_RULER_MM_OUTPUT / INCH_CONVERSION
TARGET_RULER_PX_400_PPI = 400 * TARGET_RULER_INCH_OUTPUT
CROMATIC_SCALE_RULER_400_PPI = {
    "ppi": 400,
    "width_pixel": 2095.8,
    "width_mm": 133.1,
    "height_pixel": 627,
    "height_mm": 39.8
}

# 600 PPI
TARGET_RULER_PX_600_PPI = 600 * CHROMATIC_SCALE_RULER_INCH  # cioè 190mm a 600ppi

CROMATIC_SCALE_RULER_600_PPI = {
    "ppi": 600,
    "width_pixel": 2970,
    "width_mm": 125.7,
    "height_pixel": 881.8,
    "height_mm": 37.3
}