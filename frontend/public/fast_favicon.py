from PIL import Image
import os

input_path = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\raw_logo.jpeg"
output_path_png = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\favicon.png"
output_path_ico = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\favicon.ico"

print("Running fast background removal script...")
img = Image.open(input_path).convert("RGBA")
datas = img.getdata()

new_data = []
# The fake background is made of #ffffff (white) and #cccccc (grey) blocks
for item in datas:
    # Check if pixel is grayscale and very light (white/light grey background)
    # rgb values are similar and > 180
    r, g, b, a = item
    if r > 190 and g > 190 and b > 190 and abs(r-g) < 20 and abs(g-b) < 20:
        new_data.append((255, 255, 255, 0))
    else:
        new_data.append(item)

img.putdata(new_data)

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
print("Fast processing complete! Favicon generated.")
