#!/usr/bin/env python3
"""
Turn your photo into the ASCII portrait inside dark.svg and light.svg.

  pip install pillow numpy opencv-python-headless      (optional: rembg)
  python make_portrait.py my_photo.jpg

Run it in the folder that holds dark.svg and light.svg. Backups (.bak) are made.
Options:
  --crop X,Y,W,H   manual crop box in pixels (if auto face crop is off)
  --no-cutout      keep the original background (skip background removal)
  --dir PATH       folder with the SVGs (default: current folder)
"""
import argparse, re, shutil, sys
import numpy as np
from PIL import Image, ImageOps
import cv2

COLS, ROWS = 80, 36            # character grid (chars are ~1:2, so crop aspect is 80:72)
RAMP = " .:-=+*#%@"
FONT, TLEN, X0, Y0, DY = 8, 384, 52, 128, 9.6

def load(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.array(im)

def auto_crop(rgb, manual):
    h, w = rgb.shape[:2]
    aspect = ROWS * 2 / COLS                      # crop height / width
    if manual:
        x, y, cw, ch = manual
        return x, y, cw, ch
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = casc.detectMultiScale(gray, 1.1, 5, minSize=(max(40, w // 10),) * 2)
    if len(faces):
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        cw = fw * 3.0
        ch = cw * aspect
        x = fx + fw / 2 - cw / 2
        y = fy - fh * 0.75
        return int(x), int(y), int(cw), int(ch)
    cw = min(w, h / aspect)                       # fallback: top-centre
    return int((w - cw) / 2), 0, int(cw), int(cw * aspect)

def crop_pad(arr, box, fill=0):
    x, y, cw, ch = box
    out = np.full((ch, cw) + arr.shape[2:], fill, arr.dtype)
    sx0, sy0 = max(0, x), max(0, y)
    sx1, sy1 = min(arr.shape[1], x + cw), min(arr.shape[0], y + ch)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y:sy1 - y, sx0 - x:sx1 - x] = arr[sy0:sy1, sx0:sx1]
    return out

def cutout(rgb):
    try:
        from rembg import remove
        a = np.array(remove(Image.fromarray(rgb)))[..., 3]
        return a > 128
    except Exception:
        pass
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * .03), int(h * .02), int(w * .94), int(h * .96))
    cv2.grabCut(bgr, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), 6, cv2.GC_INIT_WITH_RECT)
    m = (mask == 1) | (mask == 3)
    return m if 0.12 < m.mean() < 0.97 else np.ones((h, w), bool)

def to_rows(rgb, mask, dark):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(2.5, (6, 6)).apply(gray)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.2).astype(np.float32) / 255
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    edge = np.clip(np.hypot(gx, gy) * 1.6, 0, 1)
    lum = gray if dark else 1 - gray
    lum = np.clip(lum, 0, .88) / .88
    v = np.clip(0.8 * lum + 0.45 * edge, 0, 1) ** 0.9
    small = cv2.resize(v, (COLS, ROWS), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.float32), (COLS, ROWS), interpolation=cv2.INTER_AREA) > 0.5
    small = np.where(m, np.maximum(small, 0.12), 0)
    idx = np.clip((small * (len(RAMP) - 1) + .5).astype(int), 0, len(RAMP) - 1)
    return ["".join(RAMP[i] for i in row) for row in idx]

def reveal(delay, dur=.3, total=8):
    d = delay / total; e = min(.999, (delay + dur) / total)
    return (f'<animate attributeName="opacity" values="0;0;1;1" '
            f'keyTimes="0;{d:.4f};{e:.4f};1" dur="{total}s" fill="freeze" begin="0s"/>')

def block(rows):
    out = ["<!--PORTRAIT_START-->"]
    for i, r in enumerate(rows):
        y = Y0 + i * DY
        out.append(f'<text class="a" style="font-size:{FONT}px" x="{X0}" y="{y:.1f}" textLength="{TLEN}" '
                   f'lengthAdjust="spacing" xml:space="preserve">{r}{reveal(.4 + i * .03)}</text>')
    out.append("<!--PORTRAIT_END-->")
    return "\n".join(out)

def patch(path, rows):
    s = open(path, encoding="utf-8").read()
    pat = re.compile(r"<!--PORTRAIT_START-->.*?<!--PORTRAIT_END-->", re.S)
    if not pat.search(s):
        sys.exit(f"{path}: portrait markers not found. Use the dark.svg / light.svg I generated.")
    shutil.copy(path, path + ".bak")
    open(path, "w", encoding="utf-8").write(pat.sub(lambda _: block(rows), s))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo"); ap.add_argument("--dir", default=".")
    ap.add_argument("--crop"); ap.add_argument("--no-cutout", action="store_true")
    a = ap.parse_args()
    rgb = load(a.photo)
    manual = tuple(int(v) for v in a.crop.split(",")) if a.crop else None
    box = auto_crop(rgb, manual)
    rgb_c = crop_pad(rgb, box)
    mask = np.ones(rgb_c.shape[:2], bool) if a.no_cutout else cutout(rgb_c)
    for name, dark in (("dark", True), ("light", False)):
        patch(f"{a.dir}/{name}.svg", to_rows(rgb_c, mask, dark))
    print("done: dark.svg and light.svg updated (backups saved as .bak)")

if __name__ == "__main__":
    main()
