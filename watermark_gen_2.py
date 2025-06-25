import os
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor

input_folder = "example/input"
output_folder = "example/output"
os.makedirs(output_folder, exist_ok=True)

def get_font(font_size):
    try:
        return ImageFont.truetype("Montserrat-Light.ttf", font_size)
    except OSError:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except OSError:
            return ImageFont.load_default()

exts = [".jpg", ".jpeg", ".png", ".webp"]

def process_image(fname):
    if not any(fname.lower().endswith(ext) for ext in exts):
        return
    input_path = os.path.join(input_folder, fname)
    output_path = os.path.join(output_folder, fname.rsplit('.', 1)[0] + "_wm.png")
    image = Image.open(input_path).convert("RGBA")
    width, height = image.size
    font_size = int(height * 0.025)
    margin = int(height * 0.025)
    font = get_font(font_size)
    text_test = "text"
    bbox = font.getbbox(text_test)
    text_height = bbox[3] - bbox[1]
    watermarks = [
        ("text", (margin, height // 2), "lm", 50),
        ("text", (width - margin, margin + text_height // 2), "ra", 60),
        ("text", (margin, height - margin - text_height // 2), "la", 100),
    ]
    txt = Image.new("RGBA", image.size, (255,255,255,0))
    draw = ImageDraw.Draw(txt)
    for text, pos, anchor, alpha in watermarks:
        text_color = (255, 255, 255, alpha)
        draw.text(pos, text, font=font, fill=text_color, anchor=anchor)
    watermarked = Image.alpha_composite(image, txt).convert("RGB")
    watermarked.save(output_path)

if __name__ == "__main__":
    files = [fname for fname in os.listdir(input_folder) if any(fname.lower().endswith(ext) for ext in exts)]
    num_workers = 20
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_image, files)
