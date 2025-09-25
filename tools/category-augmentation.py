import os
import json
import numpy as np
import cv2
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

CLASS_NAMES = [
    "background", "bare land", "grass", "pavement", "road", "tree", "water",
    "agriculture land", "building", "forest land", "barren land", "urban land",
    "large vehicle", "swimming pool", "helicopter", "bridge",
    "plane", "ship", "soccer ball field", "basketball court",
    "ground track field", "small vehicle", "baseball diamond",
    "tennis court", "roundabout", "storage tank", "harbor",
    "container crane", "airport", "helipad", "chimney",
    "expressway service area", "expressway toll station", "dam",
    "golf field", "overpass", "stadium", "train station",
    "vehicle", "windmill", 'A220','A321','A330','A350','ARJ21', 'Boeing737','Boeing747','Boeing777','Boeing787',
    'bus', 'C919', 'cargo truck', 'dry cargo ship','dump truck','engineering ship','excavator','fishing boat',
    'intersection','liquid cargo ship','motorboat', 'passenger ship', 'tractor','trailer','truck tractor','tugboat',
    'van','warship'
]

def load_mask(mask_path):
    """ 读取单通道语义分割标签（支持 .png, .npy） """
    ext = os.path.splitext(mask_path)[-1].lower()
    
    if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif'] :
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    elif ext == '.npy':
        mask = np.load(mask_path)
    else:
        print(mask_path)
        raise ValueError(f"不支持的格式: {ext}")
    
    return mask

def copy_image_async(src, dst):
    """ 异步拷贝图像 """
    if not os.path.exists(dst):
        shutil.copy(src, dst)

def save_category_masks_and_images(mask, image_path, mask_filename, category_masks_dir, category_images_dir):
    """ 存储 category masks 和 category images，不创建子目录 """
    os.makedirs(category_masks_dir, exist_ok=True)
    os.makedirs(category_images_dir, exist_ok=True)

    base_name = os.path.splitext(mask_filename)[0]
    category_masks = []

    unique_labels = np.unique(mask)
    with ThreadPoolExecutor(max_workers=4) as executor:
        for label in unique_labels:
            if label == 0:
                continue

            class_name = CLASS_NAMES[label].replace("-", " ")
            category_mask_filename = f"{base_name}_{class_name.replace(' ', '_')}.png"

            category_mask_path = os.path.join(category_masks_dir, category_mask_filename)
            category_image_path = os.path.join(category_images_dir, category_mask_filename)

            if not os.path.exists(category_mask_path):
                category_mask = np.where(mask == label, label, 0).astype(np.uint8)
                cv2.imwrite(category_mask_path, category_mask)

            if not os.path.exists(category_image_path):
                executor.submit(copy_image_async, image_path, category_image_path)

            category_masks.append({
                "file_name": category_image_path.replace("\\", "/"),
                "mask_name": category_mask_path.replace("\\", "/"),
                "text": f"A satellite image of {class_name}"
            })

    return category_masks

def process_mask(mask_filename, image_dir, mask_dir, category_masks_dir, category_images_dir):
    """ 处理单个 mask 文件，生成 JSON 记录，包括 category masks 和 category images """
    mask_path = os.path.join(mask_dir, mask_filename)
    mask = load_mask(mask_path)
    unique_labels = np.unique(mask)

    if len(unique_labels) == 1 and unique_labels[0] == 0:
        return None, None

    category_names = [CLASS_NAMES[label].replace("-", " ") for label in unique_labels if label > 0 and label < len(CLASS_NAMES)]
    text = f"A satellite image of {', '.join(category_names)}"

    base_name = os.path.splitext(mask_filename)[0]
    image_filename = base_name.replace("mask", "image") + ".png"
    image_path = os.path.join(image_dir, image_filename)

    category_masks = save_category_masks_and_images(mask, image_path, mask_filename, category_masks_dir, category_images_dir)

    return {
        "file_name": image_path.replace("\\", "/"),
        "mask_name": mask_path.replace("\\", "/"),
        "text": text,
        "category_masks": [cm["mask_name"] for cm in category_masks]
    }, category_masks  # 返回单类别 JSON 记录

def load_existing_jsonl(output_jsonl):
    """ 读取已存在的 JSONL 文件，避免重复处理 """
    if not os.path.exists(output_jsonl):
        return set()
    
    existing_files = set()
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                existing_files.add(entry["file_name"])
            except json.JSONDecodeError:
                continue

    return existing_files

def generate_jsonl(image_dir, mask_dir, category_masks_dir, category_images_dir, output_jsonl, max_workers=8):
    """ 读取 mask，使用多线程加速，并**追加单类别 JSON 记录** """
    existing_files = load_existing_jsonl(output_jsonl)
    mask_filenames = [
        f for f in os.listdir(mask_dir)
        if f.lower().endswith((".png", ".npy")) and f != ".DS_Store"
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor, open(output_jsonl, "a", encoding="utf-8") as f:
        futures = {
            executor.submit(process_mask, mask_filename, image_dir, mask_dir, category_masks_dir, category_images_dir): mask_filename
            for mask_filename in mask_filenames
            if os.path.join(image_dir, mask_filename.replace("mask", "image") + ".png") not in existing_files
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing masks", unit="mask"):
            result, category_masks = future.result()

            if result is not None:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            if category_masks is not None:
                for category in category_masks:
                    f.write(json.dumps(category, ensure_ascii=False) + "\n")

    print(f"✅ JSONL file updated: {output_jsonl}")

if __name__ == '__main__':
    # 示例使用
    image_directory = "./data/EarthSynth-180K/train/images"   # 替换成你的图像目录
    mask_directory = "./data/EarthSynth-180K/train/masks"     # 替换成你的 mask 目录
    category_masks_directory = "./data/EarthSynth-180K/train/category_masks"  # 单类别 mask 存放目录
    category_images_directory = "./data/EarthSynth-180K/train/category_images"  # 单类别对应的图像存放目录
    output_jsonl_path = "./data/EarthSynth-180K/train/train.jsonl"  # 输出 JSONL 文件

    generate_jsonl(image_directory, mask_directory, category_masks_directory, category_images_directory, output_jsonl_path, max_workers=8)
