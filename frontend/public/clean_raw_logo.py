from PIL import Image
from collections import deque
import sys

input_path = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\raw_logo.jpeg"
output_path = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\clean_raw_logo.png"

print("Processing raw_logo.jpeg to remove background entirely...")
img = Image.open(input_path).convert("RGBA")
width, height = img.size
pixels = img.load()

def is_bg(r, g, b):
    # Match white or light grey checkerboard
    return r > 180 and g > 180 and b > 180 and abs(r-g) < 25 and abs(g-b) < 25

# BFS Flood Fill from all edges
visited = set()
queue = deque()

for x in range(width):
    queue.append((x, 0))
    queue.append((x, height - 1))
for y in range(height):
    queue.append((0, y))
    queue.append((width - 1, y))

bg_removed_count = 0
while queue:
    x, y = queue.popleft()
    if (x, y) in visited: continue
    visited.add((x, y))
    
    r, g, b, a = pixels[x, y]
    if is_bg(r, g, b):
        pixels[x, y] = (255, 255, 255, 0) # Make transparent
        bg_removed_count += 1
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                queue.append((nx, ny))

print(f"Removed {bg_removed_count} background pixels.")

# Do NOT crop. We want the full image including text.
img.save(output_path, format="PNG")
print("Saved clean_raw_logo.png")
