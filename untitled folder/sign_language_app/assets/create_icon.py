"""
create_icon.py  —  Generate SLRS.icns (macOS) and SLRS.ico (Windows)

Requires: Pillow  (pip install pillow)

Usage:
    cd sign_language_app/assets
    python create_icon.py
"""

import os
import struct
import subprocess
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow is required:  pip install pillow")
    sys.exit(1)

HERE = Path(__file__).parent

# ─── Design ───────────────────────────────────────────────────────────────────
SIZE         = 1024
BG_DEEP      = (10,  10,  20,  255)   # very dark navy
ACCENT_BLUE  = (0,  122, 255,  255)   # Apple blue
ACCENT_LIGHT = (64, 160, 255,  200)
WHITE        = (255, 255, 255, 230)


def draw_icon(size: int) -> Image.Image:
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d    = ImageDraw.Draw(img)
    pad  = size * 0.06

    # Background rounded square
    d.rounded_rectangle(
        [pad, pad, size - pad, size - pad],
        radius=size * 0.22,
        fill=BG_DEEP,
    )

    # Outer accent ring
    ring_w = max(3, size // 40)
    d.ellipse(
        [size * 0.10, size * 0.10, size * 0.90, size * 0.90],
        outline=ACCENT_BLUE,
        width=ring_w,
    )

    # Hand silhouette — simplified palm + 5 finger rectangles
    cx, cy = size / 2, size / 2
    # Palm
    pw, ph = size * 0.26, size * 0.26
    d.rounded_rectangle(
        [cx - pw / 2, cy - ph * 0.1, cx + pw / 2, cy + ph],
        radius=size * 0.04,
        fill=ACCENT_BLUE,
    )
    # Fingers (index, middle, ring, pinky, thumb)
    finger_w  = size * 0.07
    finger_gap = size * 0.025
    finger_tops = [
        cx - pw / 2 + finger_gap * 0.5,                   # index
        cx - pw / 2 + finger_w + finger_gap * 1.5,        # middle
        cx - pw / 2 + finger_w * 2 + finger_gap * 2.5,   # ring
        cx - pw / 2 + finger_w * 3 + finger_gap * 3.5,   # pinky
    ]
    heights = [size * 0.32, size * 0.34, size * 0.30, size * 0.24]
    for fx, fh in zip(finger_tops, heights):
        d.rounded_rectangle(
            [fx, cy - ph * 0.1 - fh, fx + finger_w, cy - ph * 0.1 + size * 0.02],
            radius=size * 0.03,
            fill=ACCENT_BLUE,
        )
    # Thumb (left side, angled slightly)
    tx = cx - pw / 2 - finger_w * 0.6
    d.rounded_rectangle(
        [tx, cy + ph * 0.05, tx + finger_w, cy + ph * 0.05 + size * 0.20],
        radius=size * 0.03,
        fill=ACCENT_LIGHT,
    )

    # "SLRS" label at bottom
    label_y = size * 0.80
    label_h = size * 0.10
    d.rectangle([size * 0.15, label_y, size * 0.85, label_y + label_h], fill=(0, 0, 0, 0))
    # Draw text character by character (no external font needed)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", int(size * 0.09))
    except Exception:
        font = ImageFont.load_default()
    text = "SLRS"
    bbox = d.textbbox((0, 0), text, font=font)
    tw   = bbox[2] - bbox[0]
    d.text(((size - tw) / 2, label_y), text, fill=WHITE, font=font)

    return img


def save_icns(img: Image.Image, out_path: Path) -> None:
    """Save a 1024×1024 RGBA image as .icns using macOS iconutil."""
    iconset = HERE / "icon.iconset"
    iconset.mkdir(exist_ok=True)

    sizes = [16, 32, 64, 128, 256, 512, 1024]
    for s in sizes:
        resized = img.resize((s, s), Image.LANCZOS)
        resized.save(iconset / f"icon_{s}x{s}.png")
        if s <= 512:   # @2x variants
            resized2 = img.resize((s * 2, s * 2), Image.LANCZOS)
            resized2.save(iconset / f"icon_{s}x{s}@2x.png")

    result = subprocess.run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(out_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[WARN] iconutil failed: {result.stderr.strip()}")
        print("       .icns not created — build will proceed without icon.")
    else:
        print(f"[✓] {out_path.name} created")

    # Clean up iconset directory
    import shutil
    shutil.rmtree(iconset, ignore_errors=True)


def save_ico(img: Image.Image, out_path: Path) -> None:
    """Save multi-resolution .ico for Windows."""
    ico_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    frames    = [img.resize(s, Image.LANCZOS).convert("RGBA") for s in ico_sizes]
    frames[0].save(
        str(out_path),
        format="ICO",
        sizes=ico_sizes,
        append_images=frames[1:],
    )
    print(f"[✓] {out_path.name} created")


if __name__ == "__main__":
    print("Generating SLRS icon...")

    icon = draw_icon(SIZE)

    # Save PNG preview
    png_path = HERE / "SLRS_icon.png"
    icon.save(str(png_path))
    print(f"[✓] {png_path.name} saved (preview)")

    # .icns (macOS)
    if sys.platform == "darwin":
        save_icns(icon, HERE / "SLRS.icns")
    else:
        print("[i] Skipping .icns — run this script on macOS to generate it.")

    # .ico (Windows — works on any platform)
    save_ico(icon, HERE / "SLRS.ico")

    print("\nDone.  Icon files are in:  assets/")
