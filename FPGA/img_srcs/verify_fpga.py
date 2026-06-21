import sys
import os
import argparse
import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("ERROR: Pillow not installed. Run:  pip install pillow numpy")
    sys.exit(1)

# ---- config: must match fire_detect.v ----
WIDTH, HEIGHT = 64, 64
SCALE = 8
Y_MIN    = 120
CR_MIN   = 150
CB_MAX   = 120
CRCB_GAP = 20


def ask_filename(value, prompt, must_exist=True):
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


def ask_int(prompt):
    while True:
        s = input(prompt).strip()
        if s == "":
            print("  please enter a whole number")
            continue
        try:
            v = int(s)
            if v < 0:
                print("  value can't be negative, try again")
                continue
            return v
        except ValueError:
            print("  not a valid integer, try again")


def load_yuv(mem_path):
    try:
        with open(mem_path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except Exception as e:
        print(f"ERROR: could not read '{mem_path}': {e}")
        sys.exit(1)
    if len(lines) != WIDTH * HEIGHT:
        print(f"ERROR: '{mem_path}' has {len(lines)} pixels, expected {WIDTH*HEIGHT}.")
        sys.exit(1)
    try:
        words = [int(ln, 16) for ln in lines]
    except ValueError as e:
        print(f"ERROR: '{mem_path}' contains a non-hex line: {e}")
        sys.exit(1)
    Y  = np.array([(w >> 16) & 0xFF for w in words]).reshape(HEIGHT, WIDTH)
    Cb = np.array([(w >> 8)  & 0xFF for w in words]).reshape(HEIGHT, WIDTH)
    Cr = np.array([ w        & 0xFF for w in words]).reshape(HEIGHT, WIDTH)
    return Y, Cb, Cr


def compute_golden(Y, Cb, Cr):
    is_fire = (Y > Y_MIN) & (Cr > CR_MIN) & (Cb < CB_MAX) & (Cr > Cb + CRCB_GAP)
    count = int(is_fire.sum())
    ys, xs = np.where(is_fire)
    return count, int(xs.sum()), int(ys.sum())


def yuv_to_rgb_image(Y, Cb, Cr):
    Yf, Cbf, Crf = Y.astype(float), Cb.astype(float), Cr.astype(float)
    R = Yf + 1.402   * (Crf - 128)
    G = Yf - 0.344136*(Cbf - 128) - 0.714136*(Crf - 128)
    B = Yf + 1.772   * (Cbf - 128)
    rgb = np.clip(np.stack([R, G, B], axis=-1), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB").resize((WIDTH*SCALE, HEIGHT*SCALE), Image.NEAREST)


def draw_crosshair(draw, cx, cy, color, size=18, width=2):
    px, py = cx * SCALE + SCALE // 2, cy * SCALE + SCALE // 2
    draw.line([(px - size, py), (px + size, py)], fill=color, width=width)
    draw.line([(px, py - size), (px, py + size)], fill=color, width=width)


def main():
    ap = argparse.ArgumentParser(description="Verify FPGA numbers vs golden, render image with both centroids.")
    ap.add_argument("mem", nargs="?", help=".mem file (default: image.mem)")
    ap.add_argument("-o", "--out", default=None, help="output PNG (default: verify_result_<memname>.png)")
    args = ap.parse_args()

    mem_path = ask_filename(args.mem or "image.mem",
                            "Reference .mem filename [image.mem]: ", must_exist=True)

    # default output name derived from the .mem filename
    if args.out:
        out_path = args.out
    else:
        mem_dir  = os.path.dirname(mem_path)
        mem_base = os.path.splitext(os.path.basename(mem_path))[0]
        out_name = f"verify_result_{mem_base}.png"
        out_path = os.path.join(mem_dir, out_name) if mem_dir else out_name

    Y, Cb, Cr = load_yuv(mem_path)
    g_count, g_sum_x, g_sum_y = compute_golden(Y, Cb, Cr)

    print("\nEnter the values your FPGA reported:")
    fpga_count = ask_int("  count = ")
    fpga_sum_x = ask_int("  sum_x = ")
    fpga_sum_y = ask_int("  sum_y = ")

    print("\n---- COMPARISON ----")
    print(f"{'':10} {'FPGA':>10} {'GOLDEN':>10} {'RESULT':>14}")
    rows = [("count", fpga_count, g_count),
            ("sum_x", fpga_sum_x, g_sum_x),
            ("sum_y", fpga_sum_y, g_sum_y)]
    all_ok = True
    for name, fpga, gold in rows:
        ok = (fpga == gold)
        all_ok &= ok
        flag = "OK" if ok else f"OFF by {fpga - gold:+d}"
        print(f"{name:10} {fpga:>10} {gold:>10} {flag:>14}")

    fpga_centroid = (fpga_sum_x // fpga_count, fpga_sum_y // fpga_count) if fpga_count > 0 else None
    gold_centroid = (g_sum_x // g_count, g_sum_y // g_count) if g_count > 0 else None

    if fpga_centroid and gold_centroid:
        print(f"\ncentroid   FPGA={fpga_centroid}   GOLDEN={gold_centroid}")
    elif g_count == 0:
        print("\nNOTE: golden count is 0 — this image has no fire pixels.")

    print("\n>>> " + ("ALL MATCH — FPGA agrees with golden reference"
                      if all_ok else
                      "MISMATCH — see the OFF rows above"))

    if not all_ok:
        print("\nDiagnosis:")
        if fpga_count != g_count:
            print(" - count off => threshold mismatch (fire_detect.v must be")
            print("   Y>120, Cr>150, Cb<120, Cr>Cb+20) or a frame-boundary/reset issue.")
        elif fpga_sum_x != g_sum_x or fpga_sum_y != g_sum_y:
            print(" - count matches but sums don't => 1-CLOCK ALIGNMENT bug.")
            print("   Check x_d/y_d/valid_d delay registers in top.v.")

    # ---- render with crosshairs ----
    img = yuv_to_rgb_image(Y, Cb, Cr)
    draw = ImageDraw.Draw(img)
    if gold_centroid:
        draw_crosshair(draw, gold_centroid[0], gold_centroid[1], (255, 0, 0))   # red = golden
    if fpga_centroid:
        draw_crosshair(draw, fpga_centroid[0], fpga_centroid[1], (0, 255, 0))   # green = FPGA

    try:
        img.save(out_path)
        print(f"\nSaved {out_path}")
        print("  RED crosshair   = golden (computed-from-image) centroid")
        print("  GREEN crosshair = FPGA-reported centroid")
        if fpga_centroid and gold_centroid:
            print("  (overlapping => perfect agreement; separated => that gap is your error)")
    except Exception as e:
        print(f"ERROR: could not save image '{out_path}': {e}")


if __name__ == "__main__":
    main()
