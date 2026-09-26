from PIL import Image
from collections import deque
import sys

input_path = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\raw_logo.jpeg"
output_path = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\raw_logo_cropped.png"

print("Processing raw_logo.jpeg...")
img = Image.open(input_path).convert("RGBA")
width, height = img.size

pixels = img.load()

def is_bg(r, g, b):
    # The checkerboard is white/grey
    # We consider light pixels (r,g,b > 180) that are greyish (low color variance) as background
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

bg_removed_count = 0
while queue:
    x, y = queue.popleft()
    
    if (x, y) in visited:
        continue
        
    visited.add((x, y))
    
    r, g, b, a = pixels[x, y]
    
    if is_bg(r, g, b):
        # Turn it transparent
        pixels[x, y] = (255, 255, 255, 0)
        bg_removed_count += 1
        
        # Add neighbors
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if (nx, ny) not in visited:
                    queue.append((nx, ny))

print(f"Removed {bg_removed_count} background pixels.")

# Tightly crop the remaining non-transparent pixels (the globe itself)
bbox = img.getbbox()
if bbox:
    print(f"Cropping to globe bounds: {bbox}")
    img = img.crop(bbox)

# Make it a perfect square with a transparent background
new_width, new_height = img.size
max_dim = max(new_width, new_height)
# Adding a small 10px padding so it breathes well in the UI
pad = 10
padded_dim = max_dim + (pad * 2)

square_img = Image.new('RGBA', (padded_dim, padded_dim), (0, 0, 0, 0))
offset = (pad + (max_dim - new_width) // 2, pad + (max_dim - new_height) // 2)
square_img.paste(img, offset)

# Save
square_img.save(output_path, format="PNG")
print("Saved transparent tightly-cropped logo to raw_logo_cropped.png")
