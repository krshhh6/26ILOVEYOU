from PIL import Image
import sys

input_path = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\new_favicon.png"
output_path_png = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\favicon.png"
output_path_ico = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\favicon.ico"

print("Resizing new_favicon.png...")
img = Image.open(input_path).convert("RGBA")

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
