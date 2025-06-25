import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor

in_dir = "example"
out_img_dir = "example"
out_mask_dir = "example"
font_path = "example"

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)

WATERMARK_TEXT = "text"
WATERMARK_COLOR = (247, 52, 152)
TEXT_COLOR = (255, 255, 255)

def process_one(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        min_side = min(w, h)
        max_side = max(w, h)
        wm_width = int(max_side * np.random.uniform(0.20, 0.25))
        font_size = 10
        for fs in range(12, 120):
            font = ImageFont.truetype(font_path, fs)
            bbox = font.getbbox(WATERMARK_TEXT)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad_x, pad_y = int(fs * 0.38), int(fs * 0.15)
            box_w, box_h = text_w + 2 * pad_x, text_h + 2 * pad_y
            if box_w >= wm_width:
                font_size = fs
                break
        font = ImageFont.truetype(font_path, font_size)
        bbox = font.getbbox(WATERMARK_TEXT)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad_x, pad_y = int(font_size * 0.38), int(font_size * 0.15)
        box_w, box_h = text_w + 2 * pad_x, text_h + 2 * pad_y
        corners = [
            (0, 0),
            (w - box_w, 0),
            (0, h - box_h),
            (w - box_w, h - box_h)
        ]
        x0, y0 = corners[np.random.randint(0, 4)]
        out_img = img.copy()
        draw = ImageDraw.Draw(out_img)
        draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=WATERMARK_COLOR)
        tx = x0 + (box_w - text_w) // 2 - bbox[0]
        ty = y0 + (box_h - text_h) // 2 - bbox[1]
        draw.text((tx, ty), WATERMARK_TEXT, font=font, fill=TEXT_COLOR)
        mask = Image.new("L", (w, h), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=255)
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_img.save(os.path.join(out_img_dir, base + ".png"))
        mask.save(os.path.join(out_mask_dir, base + "_mask.png"))
    except Exception:
        pass

def already_processed(fname):
    base = os.path.splitext(fname)[0]
    img_done = os.path.exists(os.path.join(out_img_dir, base + ".png"))
    mask_done = os.path.exists(os.path.join(out_mask_dir, base + "_mask.png"))
    return img_done and mask_done

files = [
    os.path.join(in_dir, f)
    for f in os.listdir(in_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png")) and not already_processed(f)
]

def main():
    n_threads = 16
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        list(tqdm(executor.map(process_one, files), total=len(files), desc="Processing (threaded)"))

if __name__ == "__main__":
    main()
