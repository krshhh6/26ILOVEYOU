from PIL import Image

input_path = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\new_logo.jpg"
output_path = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\cropped_logo.png"

print("Running simple crop on new_logo.jpg...")
img = Image.open(input_path).convert("RGBA")
width, height = img.size

# The user wants to remove "the white thing in the down of the picture".
# The previous crop at Y=1331 successfully removed the text block.
# Let's crop at a conservative Y value to just keep the logo.
# We will scan from the bottom up. We find the first row that is NOT purely white/light-grey.
# The text block is on a white background.

pixels = img.load()

# Find where the white block starts by scanning from the middle downwards.
# We look for a continuous block of white rows.
crop_y = int(height * 0.7) # default

for y in range(int(height * 0.5), height):
    # check if the row is entirely white/light grey
    is_white_row = True
    for x in range(width):
        r, g, b, a = pixels[x, y]
        if not (r > 240 and g > 240 and b > 240):
            is_white_row = False
            break
    if is_white_row:
        # If we find a completely white row, this might be the start of the white block!
        # Let's verify by checking the next 10 rows
        verify_count = 0
        for ny in range(y, min(y+10, height)):
            row_white = True
            for nx in range(width):
                nr, ng, nb, na = pixels[nx, ny]
                if not (nr > 240 and ng > 240 and nb > 240):
                    row_white = False
                    break
            if row_white:
                verify_count += 1
        
        if verify_count > 5:
            crop_y = y
            break

print(f"Cropping at Y={crop_y}")
img = img.crop((0, 0, width, crop_y))

# Save
img.save(output_path, format="PNG")
print("Saved to cropped_logo.png")
