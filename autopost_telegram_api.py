from pyrogram import Client
import os
import re
import random

API_ID = 123456
API_HASH = "example_api_hash"

ACCOUNTS = [
    ("user1", "@example_channel", "eng", "ExampleLabel", "ExampleHashtag"),
]

FOOTERS = {
    "ru": "example",
    "eng": "example"
}

OUT_FOLDER = "example_photos"

os.makedirs(OUT_FOLDER, exist_ok=True)

def get_sent_ids_file(username, tg_channel):
    safe_channel = tg_channel.replace("@", "")
    return os.path.join(OUT_FOLDER, f"sent_ids_{username}_{safe_channel}.txt")

def load_sent_ids(username, tg_channel):
    path = get_sent_ids_file(username, tg_channel)
    if not os.path.exists(path):
        return set()
    with open(path, "r") as f:
        return set(x.strip() for x in f)

def save_sent_id(username, tg_channel, pid):
    path = get_sent_ids_file(username, tg_channel)
    with open(path, "a") as f:
        f.write(pid + "\n")

def clean_description(desc):
    return re.sub(r"(#[\wа-яА-ЯёЁ_]+[\s]*)", "", desc, flags=re.UNICODE).strip()

def send_photo_to_telegram(photo_path, caption=None, tg_channel=None):
    try:
        with Client("example_session", api_id=API_ID, api_hash=API_HASH) as app:
            app.send_photo(
                chat_id=tg_channel,
                photo=photo_path,
                caption=caption or "",
            )
    except Exception:
        pass

def process_accounts():
    for username, tg_channel, lang, label, hashtag in ACCOUNTS:
        sent_ids = load_sent_ids(username, tg_channel)
        user_folder = os.path.join(OUT_FOLDER, username)
        if not os.path.isdir(user_folder):
            continue
        files = [f for f in os.listdir(user_folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
        files = [f for f in files if os.path.splitext(f)[0] not in sent_ids]
        if not files:
            continue
        fname = random.choice(files)
        photo_id = os.path.splitext(fname)[0]
        path = os.path.join(user_folder, fname)
        desc_path_txt = os.path.splitext(path)[0] + ".txt"
        desc_path_description = os.path.splitext(path)[0] + ".description"
        description = ""
        if os.path.exists(desc_path_txt):
            with open(desc_path_txt, "r", encoding="utf-8") as f:
                description = f.read().strip()
        elif os.path.exists(desc_path_description):
            with open(desc_path_description, "r", encoding="utf-8") as f:
                description = f.read().strip()
        desc = clean_description(description.strip())
        footer = FOOTERS.get(lang, "")
        top_line = f"[{label}](https://t.me/example)  #{hashtag}"
        if desc:
            caption = f"{top_line}\n\n{desc}\n\n{footer}"
        else:
            caption = f"{top_line}\n\n{footer}"
        send_photo_to_telegram(path, caption=caption, tg_channel=tg_channel)
        save_sent_id(username, tg_channel, photo_id)

if __name__ == "__main__":
    process_accounts()
