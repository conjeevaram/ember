import sys
import os
import argparse
import numpy as np

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not installed. Run:  pip install pillow numpy")
    sys.exit(1)

# ---- config: must match fire_detect.v ----
WIDTH, HEIGHT = 64, 64
SCALE = 8                        # preview upscale (64 -> 512) so it's visible
Y_MIN    = 120
CR_MIN   = 150
CB_MAX   = 120
CRCB_GAP = 20


def prompt_if_missing(value, prompt, must_exist=False):
    while True:
        path = value if value else input(prompt).strip().strip('"').strip("'")
        if not path:
            print("  (no filename entered, try again)")
            value = None
            continue
        if must_exist and not os.path.isfile(path):
            print(f"  ERROR: file not found: {path}")
            value = None
            continue
        return path


def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        try:
            os.makedirs(d, exist_ok=True)
            print(f"(created directory {d})")
        except Exception as e:
            print(f"ERROR: could not create directory '{d}': {e}")
            sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="Convert a PNG to a 64x64 YCbCr .mem file (+ compressed preview).")
    ap.add_argument("input",  nargs="?", help="input PNG path")
    ap.add_argument("output", nargs="?", help="output .mem path (default: image.mem)")
    args = ap.parse_args()

    in_path  = prompt_if_missing(args.input,  "Input PNG filename: ", must_exist=True)
    out_path = prompt_if_missing(args.output or "image.mem", "Output .mem filename [image.mem]: ")

    # ---- load + downscale ----
    try:
        img = Image.open(in_path).convert("RGB").resize((WIDTH, HEIGHT))
    except Exception as e:
        print(f"ERROR: could not open/process image '{in_path}': {e}")
        sys.exit(1)

    rgb = np.array(img).astype(int)
    if rgb.shape != (HEIGHT, WIDTH, 3):
        print(f"ERROR: unexpected image shape {rgb.shape}, expected ({HEIGHT},{WIDTH},3)")
        sys.exit(1)

    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # RGB -> YCbCr (ITU-R BT.601)
    Y  = ( 0.299*R + 0.587*G + 0.114*B).clip(0, 255).astype(int)
    Cb = (-0.168736*R - 0.331264*G + 0.5*B + 128).clip(0, 255).astype(int)
    Cr = ( 0.5*R - 0.418688*G - 0.081312*B + 128).clip(0, 255).astype(int)

    # ---- write .mem ----
    ensure_dir(out_path)
    try:
        with open(out_path, "w") as f:
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    word = (Y[y, x] << 16) | (Cb[y, x] << 8) | Cr[y, x]
                    f.write(f"{word:06x}\n")
    except Exception as e:
        print(f"ERROR: could not write '{out_path}': {e}")
        sys.exit(1)
    print(f"\nWrote {out_path}  ({WIDTH*HEIGHT} pixels, {WIDTH}x{HEIGHT})")

    # ---- save 'compressed_' preview: exactly what the FPGA sees ----
    Yf, Cbf, Crf = Y.astype(float), Cb.astype(float), Cr.astype(float)
    Rr = Yf + 1.402   * (Crf - 128)
    Gg = Yf - 0.344136*(Cbf - 128) - 0.714136*(Crf - 128)
    Bb = Yf + 1.772   * (Cbf - 128)
    recon = np.clip(np.stack([Rr, Gg, Bb], axis=-1), 0, 255).astype(np.uint8)
    preview = Image.fromarray(recon, "RGB").resize((WIDTH*SCALE, HEIGHT*SCALE), Image.NEAREST)

    out_dir   = os.path.dirname(out_path)
    base      = os.path.splitext(os.path.basename(in_path))[0]
    comp_name = f"compressed_{base}.png"
    comp_path = os.path.join(out_dir, comp_name) if out_dir else comp_name
    ensure_dir(comp_path)
    try:
        preview.save(comp_path)
        print(f"Wrote {comp_path}  (exactly what the FPGA sees, upscaled {SCALE}x)")
    except Exception as e:
        print(f"WARNING: could not save compressed preview: {e}")

    # ---- golden reference (same rules as fire_detect.v) ----
    is_fire = (Y > Y_MIN) & (Cr > CR_MIN) & (Cb < CB_MAX) & (Cr > Cb + CRCB_GAP)
    count = int(is_fire.sum())

    print("\n---- GOLDEN REFERENCE (FPGA should reproduce these) ----")
    print(f"count = {count}")
    if count > 0:
        ys, xs = np.where(is_fire)
        sum_x, sum_y = int(xs.sum()), int(ys.sum())
        print(f"sum_x = {sum_x}")
        print(f"sum_y = {sum_y}")
        print(f"centroid = ({sum_x // count}, {sum_y // count})  (integer division)")
    else:
        print("WARNING: no fire pixels — this image won't trip the threshold.")
        print("         (check the image actually has bright orange/red regions)")


if __name__ == "__main__":
    main()
