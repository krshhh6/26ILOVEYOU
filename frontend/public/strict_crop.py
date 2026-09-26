from PIL import Image

input_path = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\new_logo.jpg"
output_path = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\cropped_logo.png"

print("Running strict crop on new_logo.jpg...")
img = Image.open(input_path).convert("RGB")
width, height = img.size

# We will NOT touch any transparency or pixels so we don't accidentally ruin the globe.
# We just crop off the bottom 35% where the white text box is.
crop_y = int(height * 0.65)
img = img.crop((0, 0, width, crop_y))

# Make it a square so it renders nicely in a 1:1 img tag
# Let's pick the background color from the top left pixel so the padding matches
bg_color = img.getpixel((0, 0))

new_width, new_height = img.size
max_dim = max(new_width, new_height)
square_img = Image.new('RGB', (max_dim, max_dim), bg_color)
offset = ((max_dim - new_width) // 2, (max_dim - new_height) // 2)
square_img.paste(img, offset)

# Save
square_img.save(output_path, format="PNG")
print("Saved clean, unmodified logo (cropped) to cropped_logo.png")
