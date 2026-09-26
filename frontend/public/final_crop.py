from PIL import Image
from collections import deque

input_path = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\new_logo.jpg"
output_path = r"d:\SIH-26143-OIL-Spill\SIH-26143-OIL-Spill\frontend\public\final_globe.png"

print("Running ultimate crop on new_logo.jpg...")
img = Image.open(input_path).convert("RGBA")
width, height = img.size

# 1. Crop off the bottom text completely. The globe is in the top 60%.
crop_y = int(height * 0.60)
img = img.crop((0, 0, width, crop_y))
width, height = img.size

pixels = img.load()

# 2. Flood fill from the edges to remove the white/grey background.
# We will use a generous threshold for white/light-grey.
def is_bg(r, g, b):
    # Match white or light grey checkerboard
    return r > 200 and g > 200 and b > 200 and abs(r-g) < 20 and abs(g-b) < 20

visited = set()
queue = deque()

for x in range(width):
    queue.append((x, 0))
    queue.append((x, height - 1))
for y in range(height):
    queue.append((0, y))
    queue.append((width - 1, y))

while queue:
    x, y = queue.popleft()
    if (x, y) in visited: continue
    visited.add((x, y))
    
    r, g, b, a = pixels[x, y]
    if is_bg(r, g, b):
        pixels[x, y] = (255, 255, 255, 0) # transparent
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                queue.append((nx, ny))

# 3. Tightly crop the globe
bbox = img.getbbox()
if bbox:
    img = img.crop(bbox)

# 4. Make it a perfect square
new_width, new_height = img.size
max_dim = max(new_width, new_height)
square_img = Image.new('RGBA', (max_dim, max_dim), (0, 0, 0, 0))
offset = ((max_dim - new_width) // 2, (max_dim - new_height) // 2)
square_img.paste(img, offset)

# Save
square_img.save(output_path, format="PNG")
print("Saved final_globe.png")
