from PIL import Image

from utils import *

# Percorso file immagine di input (modifica se necessario)
for image_filename in 

# Output path
OUTPUT_FILENAME = f"scaled_to_{CHROMATIC_SCALE_RULER_MM}mm.tif"
OUTPUT_PATH = IMAGES_OUTPUT_DIR / OUTPUT_FILENAME

# Lunghezza righello originale e desiderata (in mm)
original_ruler_mm = CROMATIC_SCALE_RULER_400_PPI["width_mm"]
target_ruler_mm = CHROMATIC_SCALE_RULER_MM

# Calcola fattore di scala
scale_factor = target_ruler_mm / original_ruler_mm
print(f"[INFO] Scaling image from {original_ruler_mm:.2f} mm → {target_ruler_mm} mm")
print(f"[INFO] Scale factor: {scale_factor:.4f}")

# Carica immagine
input_path = IMAGES_INPUT_DIR / INPUT_FILENAME
img = Image.open(input_path)
original_size = (img.width, img.height)
print(f"[INFO] Original image size: {original_size}")

# Applica resize
new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
resized_img = img.resize(new_size, resample=Image.LANCZOS)
print(f"[INFO] New image size: {new_size}")

# Salva immagine ridimensionata con PPI = 400
resized_img.save(OUTPUT_PATH, dpi=(400, 400))
print(f"[DONE] Saved resized image to: {OUTPUT_PATH}")
