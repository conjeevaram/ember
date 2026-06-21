import numpy as np
from PIL import Image, ImageDraw

# Read the .mem file (4096 lines, each 6 hex digits = YY Cb Cr)
with open("image.mem") as f:
    words = [int(line.strip(), 16) for line in f if line.strip()]

assert len(words) == 4096, f"expected 4096 pixels, got {len(words)}"

# Unpack YUV
Y  = np.array([(w >> 16) & 0xFF for w in words]).reshape(64, 64).astype(float)
Cb = np.array([(w >> 8)  & 0xFF for w in words]).reshape(64, 64).astype(float)
Cr = np.array([ w        & 0xFF for w in words]).reshape(64, 64).astype(float)

# YCbCr -> RGB (ITU-R BT.601)
R = Y + 1.402   * (Cr - 128)
G = Y - 0.344136*(Cb - 128) - 0.714136*(Cr - 128)
B = Y + 1.772   * (Cb - 128)
rgb = np.clip(np.stack([R, G, B], axis=-1), 0, 255).astype(np.uint8)

# Scale up 8x so 64x64 is actually visible
img = Image.fromarray(rgb, "RGB").resize((512, 512), Image.NEAREST)
img.save("image_preview.png")
print("Saved image_preview.png")

# Overlay the FPGA's detected centroid (29, 32) as a green crosshair
draw = ImageDraw.Draw(img)
cx, cy = 29 * 8, 32 * 8   # scale centroid coords to the 512px image
draw.line([(cx-15, cy), (cx+15, cy)], fill=(0,255,0), width=2)
draw.line([(cx, cy-15), (cx, cy+15)], fill=(0,255,0), width=2)
img.save("image_with_centroid.png")
print("Saved image_with_centroid.png")
