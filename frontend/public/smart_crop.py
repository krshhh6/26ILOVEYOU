from PIL import Image
import sys

input_path = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\new_logo.jpg"
output_path = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\cropped_logo.png"

print("Running smart crop on new_logo.jpg...")
img = Image.open(input_path).convert("RGBA")
width, height = img.size

pixels = img.load()

# 1. Remove background (white and light grey checkerboard)
for y in range(height):
    for x in range(width):
        r, g, b, a = pixels[x, y]
        if r > 180 and g > 180 and b > 180 and abs(r-g) < 25 and abs(g-b) < 25:
            pixels[x, y] = (0, 0, 0, 0)

# 2. Find the horizontal gap between the globe and the text
# We scan rows from the middle downwards to find a completely transparent row
# or just crop the top 65% since standard logos like this have text at the bottom 1/3.
crop_y = int(height * 0.65) # fallback

# Let's try to find the actual gap. We start scanning from y = height*0.3 to height*0.8
for y in range(int(height * 0.3), int(height * 0.8)):
    transparent_pixels = sum(1 for x in range(width) if pixels[x, y][3] == 0)
    # If the row is 98% transparent, it's a gap!
    if transparent_pixels > width * 0.98:
        crop_y = y
        break

print(f"Cropping at Y={crop_y} (Image height={height})")

# Crop just the top portion (the globe)
img = img.crop((0, 0, width, crop_y))

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

# Save
square_img.save(output_path, format="PNG")
print("Cropped logo saved to cropped_logo.png")
