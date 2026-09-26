from PIL import Image
import os

input_path = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\new_logo.jpg"
output_path_png = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\favicon.png"
output_path_ico = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\favicon.ico"

print("Running background removal on new_logo.jpg...")
img = Image.open(input_path).convert("RGBA")
width, height = img.size

# We will read pixels directly
pixels = img.load()

for y in range(height):
    for x in range(width):
        r, g, b, a = pixels[x, y]
        # Remove light grey / white checkerboard and white bottom part
        if r > 180 and g > 180 and b > 180 and abs(r-g) < 25 and abs(g-b) < 25:
            pixels[x, y] = (255, 255, 255, 0)

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

# Save high-res PNG
img_png = square_img.resize((192, 192), Image.Resampling.LANCZOS)
img_png.save(output_path_png, format="PNG")

# Save ICO
img_ico = square_img.resize((32, 32), Image.Resampling.LANCZOS)
img_ico.save(output_path_ico, format="ICO")
print("Favicon correctly generated and saved.")
