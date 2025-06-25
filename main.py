import os
import cv2
import glob
import torch
import numpy as np
import shutil
import subprocess
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}
PROXIES = {
    "http":  "----",
    "https": "----",
}
DOWNLOAD_FOLDER = "/workspace/result/lama_images/"
DOWNLOAD_VIDEO_FOLDER = "/workspace/result/dataset/"
NEURO_IMAGES_DIR = "/workspace/result/dataset/images"
NEURO_MASKS_DIR = "/workspace/result/neuro_masks"
LAMA_INPUT_DIR = "/workspace/result/lama_input"
LAMA_OUTPUT_DIR = "/workspace/result/lama_output"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_VIDEO_FOLDER, exist_ok=True)
os.makedirs(NEURO_IMAGES_DIR, exist_ok=True)
os.makedirs(NEURO_MASKS_DIR, exist_ok=True)
MAX_DOWNLOAD_THREADS = 20
VALID_EXTS = [".png", ".jpg", ".jpeg"]
VIDEO_EXTS = [".mp4", ".webm", ".avi", ".mov", ".mkv", ".MP4", ".WEBM", ".AVI", ".MOV", ".MKV"]

session = requests.Session()
session.headers.update(HEADERS)
session.proxies.update(PROXIES)

def extract_main_media_url(page_url):
    try:
        res = session.get(page_url, timeout=30)
        soup = BeautifulSoup(res.text, "html.parser")
        main_div = soup.find("div", class_="flex justify-between items-center")
        if main_div:
            a_tag = main_div.find("a", class_="uk-align-center", href=True)
            if a_tag:
                img_tag = a_tag.find("img")
                if img_tag and img_tag.get("src"):
                    media_url = urljoin(page_url, img_tag["src"])
                    return media_url, "image"
                video_tag = a_tag.find("video")
                if video_tag:
                    source = video_tag.find("source", src=True)
                    if source:
                        return urljoin(page_url, source["src"]), "video"
        media_urls = []
        for img in soup.find_all("img", src=True):
            media_urls.append(urljoin(page_url, img["src"]))
        for video in soup.find_all("video"):
            for source in video.find_all("source", src=True):
                media_urls.append(urljoin(page_url, source["src"]))
        if not media_urls:
            return None, None
        largest_url, largest_size = None, 0
        for url in media_urls:
            try:
                head = session.head(url, timeout=30, allow_redirects=True)
                size = int(head.headers.get("Content-Length", 0))
                if size > largest_size:
                    largest_size = size
                    largest_url = url
            except Exception:
                continue
        if largest_url:
            ext = os.path.splitext(largest_url)[-1].lower()
            if ext in VIDEO_EXTS:
                return largest_url, "video"
            else:
                return largest_url, "image"
        return None, None
    except Exception:
        return None, None

def download_file(url, filename, is_video=False):
    folder = DOWNLOAD_VIDEO_FOLDER if is_video else DOWNLOAD_FOLDER
    try:
        response = session.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(os.path.join(folder, filename), "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
    except Exception:
        pass

def process_page(username, page):
    url = f"------/{username}/{page}/"
    media_url, media_type = extract_main_media_url(url)
    if not media_url:
        return
    ext = os.path.splitext(media_url)[-1].lower()
    is_video = (media_type == "video") or (ext in VIDEO_EXTS)
    filename = f"{username}_{page}{ext}"
    download_file(media_url, filename, is_video=is_video)

def parse_fapello(username, start_page, end_page):
    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_THREADS) as executor:
        futures = [executor.submit(process_page, username, page)
                   for page in range(start_page, end_page + 1)]
        for _ in as_completed(futures):
            pass

def convert_images_to_png(src_folder, dst_folder):
    files = []
    for ext in VALID_EXTS + [e.upper() for e in VALID_EXTS]:
        files += glob.glob(os.path.join(src_folder, f"*{ext}"))
    files = sorted(set(files))
    os.makedirs(dst_folder, exist_ok=True)
    for path in tqdm(files, desc="Convert to PNG"):
        img = cv2.imread(path)
        if img is None:
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(dst_folder, f"{name}.png")
        cv2.imwrite(out_path, img)

def neuro_mask_inference(images_dir, masks_dir, model_path, batch_size=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.UnetPlusPlus(encoder_name="resnet50", in_channels=3, classes=1, encoder_weights=None)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    transform = A.Compose([A.Resize(256, 256), ToTensorV2()])
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    os.makedirs(masks_dir, exist_ok=True)
    def load_image(path):
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)
        orig_size = img.size
        tensor = transform(image=img_np)["image"]
        return tensor, orig_size, os.path.basename(path).rsplit(".", 1)[0]
    batch = []
    meta = []
    for path in tqdm(image_paths, desc="Infer masks"):
        tensor, size, name = load_image(path)
        batch.append(tensor)
        meta.append((size, name))
        if len(batch) == batch_size:
            tensors = torch.stack(batch).float().to(device)
            with torch.no_grad():
                preds = torch.sigmoid(model(tensors)).cpu().numpy()
            for i in range(len(preds)):
                mask = (preds[i, 0] > 0.5).astype(np.uint8) * 255
                mask_img = Image.fromarray(mask).resize(meta[i][0], resample=Image.NEAREST)
                mask_img.save(os.path.join(masks_dir, f"{meta[i][1]}_neuromask.png"))
            batch, meta = [], []
    if batch:
        tensors = torch.stack(batch).float().to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(tensors)).cpu().numpy()
        for i in range(len(preds)):
            mask = (preds[i, 0] > 0.5).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask).resize(meta[i][0], resample=Image.NEAREST)
            mask_img.save(os.path.join(masks_dir, f"{meta[i][1]}_neuromask.png"))

def prepare_lama_input_from_neuro(images_dir, masks_dir, lama_input_dir):
    if os.path.exists(lama_input_dir):
        shutil.rmtree(lama_input_dir)
    os.makedirs(lama_input_dir)
    for img_path in sorted(glob.glob(os.path.join(images_dir, "*.png"))):
        name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(masks_dir, f"{name}_neuromask.png")
        shutil.copy2(img_path, os.path.join(lama_input_dir, f"{name}.png"))
        if os.path.exists(mask_path):
            shutil.copy2(mask_path, os.path.join(lama_input_dir, f"{name}_mask.png"))

def run_lama_gpu(lama_path, model_dir, model_ckpt, input_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    cmd = [
        "python3", os.path.join(lama_path, "bin", "predict.py"),
        "refine=False",
        f"model.path={model_dir}",
        f"model.checkpoint={model_ckpt}",
        f"indir={input_dir}",
        f"outdir={output_dir}",
        "device=cpu",
        "+gpu_ids=0"
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = lama_path
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError:
        pass

def crop_inpainted_with_top_left_mask(lama_input_dir, lama_output_dir, crop_px=30, left_px=150):
    mask_paths = sorted(glob.glob(os.path.join(lama_input_dir, "*_mask.png")))
    for mask_path in mask_paths:
        mask_name = os.path.basename(mask_path)
        out_path = os.path.join(lama_output_dir, mask_name)
        if not os.path.exists(out_path):
            continue
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            continue
        top_left = mask[:crop_px, :left_px]
        white_pixels = np.sum(top_left > 128)
        if white_pixels == 0:
            continue
        img = cv2.imread(out_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        if h <= crop_px:
            continue
        cropped = img[crop_px:, :]
        cv2.imwrite(out_path, cropped)

if __name__ == "__main__":
    user = input("test1").strip()
    start = int(input("test2").strip())
    end = int(input("test3").strip())
    parse_fapello(user, start, end)
    convert_images_to_png(DOWNLOAD_FOLDER, NEURO_IMAGES_DIR)
    neuro_mask_inference(NEURO_IMAGES_DIR, NEURO_MASKS_DIR, "123.pth")
    prepare_lama_input_from_neuro(NEURO_IMAGES_DIR, NEURO_MASKS_DIR, LAMA_INPUT_DIR)
    lama_path = "/workspace/neuro/lama"
    model_dir = "/workspace/lama_models/"
    model_ckpt = "best.ckpt"
    run_lama_gpu(lama_path, model_dir, model_ckpt, LAMA_INPUT_DIR, LAMA_OUTPUT_DIR)
    crop_inpainted_with_top_left_mask(LAMA_INPUT_DIR, LAMA_OUTPUT_DIR, crop_px=30, left_px=150)
