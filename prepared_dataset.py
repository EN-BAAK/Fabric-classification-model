import os
import glob
import shutil

base_dir = '../dataset'
for fabric_type in os.listdir(base_dir):
    fabric_path = os.path.join(base_dir, fabric_type)
    if not os.path.isdir(fabric_path):
        continue

    for subfolder in os.listdir(fabric_path):
        subfolder_path = os.path.join(fabric_path, subfolder)
        if os.path.isdir(subfolder_path):
            for img_file in os.listdir(subfolder_path):
                src = os.path.join(subfolder_path, img_file)
                dst = os.path.join(fabric_path, f"{subfolder}_{img_file}")
                shutil.move(src, dst)
            os.rmdir(subfolder_path)

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')

for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue  

    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    for txt_file in txt_files:
        os.remove(txt_file)
        print(f"Deleted: {txt_file}")

    image_files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(image_extensions)
    ]

    image_files.sort()

print("Cleanup complete.")
