from PIL import Image
from collections import deque

input_path = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\new_logo.jpg"
output_path = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\cropped_logo.png"

print("Running smart flood-fill crop...")
img = Image.open(input_path).convert("RGBA")
width, height = img.size

# We know the bottom is the white text box, crop that off first
crop_y = int(height * 0.65)
img = img.crop((0, 0, width, crop_y))
width, height = img.size

pixels = img.load()

# The background is a fake checkerboard pattern.
# Let's identify the grey color from the top left corners. 
# Usually it's white (255,255,255) and some grey.
# Let's sample a small region in the top left to find the grey color.
bg_colors = []
for y in range(20):
    for x in range(20):
        r, g, b, a = pixels[x, y]
        if (r, g, b) not in bg_colors:
            bg_colors.append((r, g, b))

print(f"Colors found in top-left 20x20: {bg_colors[:10]}")

# Instead of strict matching, any pixel that is very light gray/white (r,g,b > 180 and close to each other)
# will be considered "checkerboard background" IF it's connected to the edge!
def is_bg(r, g, b):
    # Check if pixel is a shade of light grey or white
    return r > 180 and g > 180 and b > 180 and abs(r-g) < 25 and abs(g-b) < 25

# BFS Flood Fill from all edges
visited = set()
queue = deque()

# Add all edge pixels
for x in range(width):
    queue.append((x, 0))
    queue.append((x, height - 1))
for y in range(height):
    queue.append((0, y))
    queue.append((width - 1, y))

while queue:
    x, y = queue.popleft()
    
    if (x, y) in visited:
        continue
        
    visited.add((x, y))
    
    r, g, b, a = pixels[x, y]
    
    if is_bg(r, g, b):
        # Turn it transparent
        pixels[x, y] = (255, 255, 255, 0)
        
        # Add neighbors
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if (nx, ny) not in visited:
                    queue.append((nx, ny))

# Now we have a transparent background.
# Crop transparent borders tightly
bbox = img.getbbox()
if bbox:
    img = img.crop(bbox)

# Make it a square
new_width, new_height = img.size
max_dim = max(new_width, new_height)
square_img = Image.new('RGBA', (max_dim, max_dim), (0, 0, 0, 0))
offset = ((max_dim - new_width) // 2, (max_dim - new_height) // 2)
square_img.paste(img, offset)

# Save
square_img.save(output_path, format="PNG")
print("Saved transparent flood-filled logo to cropped_logo.png")
