import sys
import os
try:
    from rembg import remove
    from PIL import Image
    import io
except ImportError:
    print("Dependencies missing. Please install them.")
    sys.exit(1)

input_path = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\raw_logo.jpeg"
output_path_png = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\favicon.png"
output_path_ico = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\favicon.ico"

print("Loading image...")
with open(input_path, 'rb') as i:
    input_data = i.read()

print("Removing background with rembg...")
output_data = remove(input_data)

print("Opening processed image in Pillow...")
img = Image.open(io.BytesIO(output_data))

# Crop transparent borders tightly
bbox = img.getbbox()
if bbox:
    img = img.crop(bbox)

# Make it a perfect square
width, height = img.size
max_dim = max(width, height)
square_img = Image.new('RGBA', (max_dim, max_dim), (0, 0, 0, 0))
offset = ((max_dim - width) // 2, (max_dim - height) // 2)
square_img.paste(img, offset)

# Save high-res PNG for manifest/apple-touch-icon
img_png = square_img.resize((192, 192), Image.Resampling.LANCZOS)
img_png.save(output_path_png, format="PNG")

# Save ICO
img_ico = square_img.resize((32, 32), Image.Resampling.LANCZOS)
img_ico.save(output_path_ico, format="ICO")
print("Successfully generated favicon.png and favicon.ico!")
