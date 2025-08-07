import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['savefig.dpi'] = 500  # 图片像素
plt.rcParams['figure.dpi'] = 500  # 分辨率
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

CLASS_NAMES = [
    # 40 classes from the dataset
    "background", "bare land", "grass", "pavement", "road", "tree", "water",
    "agriculture land", "building", "forest land", "barren land", "urban land",
    "large vehicle", "swimming pool", "helicopter", "bridge",
    "plane", "ship", "soccer ball field", "basketball court",
    "ground track field", "small vehicle", "baseball diamond",
    "tennis court", "roundabout", "storage tank", "harbor",
    "container crane", "airport", "helipad", "chimney",
    "expressway service area", "expressway toll station", "dam",
    "golf field", "overpass", "stadium", "train station",
    "vehicle", "windmill", 
    # Other classes
    'A220','A321','A330','A350','ARJ21', 'Boeing737','Boeing747','Boeing777','Boeing787', 'bus', 'C919', 'cargo truck', 'dry cargo ship','dump truck','engineering ship','excavator','fishing boat', 'intersection','liquid cargo ship','motorboat', 'passenger ship', 'tractor','trailer','truck tractor','tugboat','van','warship', "rangeland", "developed space"
]

def load_image(image_path):
    """ 读取RGB图像 """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV 读取 BGR，需要转换
    return image

def load_mask(mask_path):
    """ 读取单通道语义分割标签（支持 .png, .npy） """
    ext = os.path.splitext(mask_path)[-1].lower()
    
    if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif']:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)  # 读取单通道
        if mask.ndim == 3:  # 如果是彩色 mask，转换为单通道
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    elif ext == '.npy':  # 读取 npy 格式
        mask = np.load(mask_path)
    else:
        raise ValueError(f"不支持的格式: {ext}")
    
    return mask

def generate_colormap(num_classes=40):
    """ 生成固定的类别颜色映射 """
    np.random.seed(42)  # 保证颜色一致
    # np.random.seed(37)  # 保证颜色一致
    colormap = {i: np.random.randint(0, 255, (3,), dtype=np.uint8) for i in range(num_classes)}
    return colormap

def apply_colormap(mask, colormap):
    """ 将单通道 mask 映射为彩色 mask """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls in np.unique(mask):
        if cls in colormap:
            color_mask[mask == cls] = colormap[cls]

    return color_mask

def overlay_mask(image, mask_colored, alpha=0.5):
    """ 叠加 mask 到原图（使用透明度 alpha 控制） """
    overlay = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)
    return overlay

def visualize_and_save(image_path, mask_path, output_dir="vis_out"):
    """ 可视化语义分割结果并保存，同时显示类别颜色对应关系 """
    os.makedirs(output_dir, exist_ok=True)

    # 读取图像和mask
    image = load_image(image_path)
    mask = load_mask(mask_path)
    colormap = generate_colormap(len(CLASS_NAMES))  # 生成颜色映射
    mask_colored = apply_colormap(mask, colormap)
    overlay = overlay_mask(image, mask_colored, alpha=0.5)

    # 保存单独的mask、叠加图、原图
    mask_colored_bgr = cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(output_dir, "original_image.png"), image_bgr)
    cv2.imwrite(os.path.join(output_dir, "semantic_mask.png"), mask_colored_bgr)
    cv2.imwrite(os.path.join(output_dir, "overlay_result.png"), overlay_bgr)

    print(f"Results saved to {output_dir}")

    # 生成类别颜色图例
    unique_classes = np.unique(mask)
    legend_patches = []
    for cls in unique_classes:
        if cls < len(CLASS_NAMES):
            class_name = CLASS_NAMES[cls]
        else:
            class_name = f"Unknown ({cls})"
        
        color = np.array(colormap[cls]) / 255.0  # 归一化到 [0,1] 以适应 Matplotlib
        legend_patches.append(mpatches.Patch(color=color, label=class_name))

    # Matplotlib 显示
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    # axes[0].set_title("Original Image",fontsize=12, fontweight='bold')
    axes[0].axis("off")

    axes[1].imshow(mask_colored)
    # axes[1].set_title("Semantic Mask",fontsize=12, fontweight='bold')
    axes[1].axis("off")

    axes[2].imshow(overlay)
    # axes[2].set_title("Overlay Image",fontsize=12, fontweight='bold')
    axes[2].axis("off")

    # # Matplotlib 显示
    # fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # axes[0].imshow(image)
    # axes[0].set_title("Original Image")
    # axes[0].axis("off")

    # axes[1].imshow(mask_colored)
    # axes[1].set_title("Semantic Mask")
    # axes[1].axis("off")



    # 添加图例
    plt.legend(handles=legend_patches, fontsize=15,loc='center left', bbox_to_anchor=(1, 0.5))
    
    # 保存完整可视化结果
    visualization_path = os.path.join(output_dir, "visualization.png")
    plt.savefig(visualization_path, bbox_inches="tight")
    # plt.show()

    print(f"Visualization saved to {visualization_path}")


image_path = "path/to/image"  # 替换成你的图像路径
mask_path = "/path/to/mask"  # 替换成你的 mask 路径



visualize_and_save(image_path, mask_path)