import os
import zipfile

folder = "example/input"
output_folder = "example/output"
max_size = 1000 * 1024 * 1024

os.makedirs(output_folder, exist_ok=True)

files = [
    os.path.join(folder, f)
    for f in sorted(os.listdir(folder))
    if os.path.isfile(os.path.join(folder, f))
]

current_part = 1
current_size = 0
current_files = []
for file_path in files:
    file_size = os.path.getsize(file_path)
    if current_size + file_size > max_size and current_files:
        archive_name = os.path.join(output_folder, f"archive_{current_part}.zip")
        with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for f in current_files:
                zipf.write(f, arcname=os.path.basename(f))
        current_part += 1
        current_size = 0
        current_files = []
    current_files.append(file_path)
    current_size += file_size

if current_files:
    archive_name = os.path.join(output_folder, f"archive_{current_part}.zip")
    with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in current_files:
            zipf.write(f, arcname=os.path.basename(f))
