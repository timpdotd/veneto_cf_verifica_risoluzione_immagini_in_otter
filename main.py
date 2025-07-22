import torch
import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from utils import *
from model.SR_Script.super_resolution import SA_SuperResolution


def numpy_to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array)


def check_super_resolution_output_dir() -> Path:
    if SUPER_RESOLUTION_PAR == 2:
        output_dir = IMAGES_SR_X2_DIR
    elif SUPER_RESOLUTION_PAR == 3:
        output_dir = IMAGES_SR_X3_DIR
    elif SUPER_RESOLUTION_PAR == 4:
        output_dir = IMAGES_SR_X4_DIR
    else:
        raise ValueError("Unsupported super resolution scale. Supported values are 2, 3, or 4.")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_super_resolution_images(image_paths: list[str]):
    output_dir = check_super_resolution_output_dir()
    print(f"Super-resolution output directory: {output_dir}")
    for img_filename in tqdm(image_paths, desc="Super-resolution"):
        img_path = IMAGES_INPUT_DIR / img_filename
        img_np = np.array(Image.open(img_path).convert("RGB"))
        upscaled_image_np = model.run(img_np)
        output_img = numpy_to_image(upscaled_image_np)
        output_path = IMAGES_SR_X2_DIR / img_filename
        if output_path.exists():
            output_path.unlink()
        output_img.save(output_path)


def create_personalized_downscaling(image_paths: list[str]):
    super_resolution_dir = check_super_resolution_output_dir()
    print(f"Personalized downscaling output directory: {IMAGES_DOWNSIZED_DIR}")
    for img_filename in tqdm(image_paths, desc="Downscaling"):
        img_path = super_resolution_dir / img_filename
        try:
            requested_PPI = int(img_filename.split("_")[-1].split(".")[0])
        except ValueError:
            continue

        if requested_PPI == 400:
            chromatic_ruler = CROMATIC_SCALE_RULER_400_PPI
            target_ruler_px = TARGET_RULER_PX_400_PPI
        elif requested_PPI == 600:
            chromatic_ruler = CROMATIC_SCALE_RULER_600_PPI
            target_ruler_px = TARGET_RULER_PX_600_PPI
        else:
            continue

        original_ruler_px = SUPER_RESOLUTION_PAR * chromatic_ruler["width_pixel"]
        scale_factor = target_ruler_px / original_ruler_px

        image = Image.open(img_path)
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)

        resized_img = image.resize((new_width, new_height), resample=Image.LANCZOS)
        output_path = IMAGES_DOWNSIZED_DIR / img_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resized_img.save(output_path, dpi=(requested_PPI, requested_PPI))

"""
def create_geometrically_resized_images(image_paths: list[str]):
    for img_filename in tqdm(image_paths, desc="Geometric Resize"):
        img_path = IMAGES_INPUT_DIR / img_filename
        try:
            requested_PPI = int(img_filename.split("_")[-1].split(".")[0])
        except ValueError:
            continue

        if requested_PPI == 400:
            chromatic_ruler = CROMATIC_SCALE_RULER_400_PPI
        elif requested_PPI == 600:
            chromatic_ruler = CROMATIC_SCALE_RULER_600_PPI
        else:
            continue

        original_ruler_mm = SUPER_RESOLUTION_PAR * chromatic_ruler["width_pixel"]
        scale_factor = CHROMATIC_SCALE_RULER_MM / original_ruler_mm

        img = Image.open(img_path)
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        resized_img = img.resize(new_size, resample=Image.LANCZOS)

        output_path = IMAGES_RESIZED_DIR / f"resized_{img_filename}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resized_img.save(output_path, dpi=(requested_PPI, requested_PPI))
"""

if __name__ == "__main__":
    model = SA_SuperResolution(
        models_dir=SR_SCRIPT_MODEL_DIR,
        model_scale=SUPER_RESOLUTION_PAR,
        tile_size=128,
        gpu_id=0,
        verbosity=True,
    )

    super_resolution_dir = check_super_resolution_output_dir()
    super_resolution_dir.mkdir(parents=True, exist_ok=True)

    image_paths = os.listdir(IMAGES_INPUT_DIR)

    create_super_resolution_images(image_paths)

    sr_images = os.listdir(IMAGES_SR_X2_DIR)
    create_personalized_downscaling(sr_images)

    """
    image_paths = os.listdir(IMAGES_INPUT_DIR)
    create_geometrically_resized_images(image_paths)
    """