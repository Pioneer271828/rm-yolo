"""
将 picture/images 中的图片和对应标注，按比例分割为训练集和验证集
输出到 picture/train 和 picture/val，不影响现有 Dataset 目录
"""

import random
import shutil
from pathlib import Path

# ===== 配置区域 =====
IMAGE_DIR   = r"F:\RM-Demo\b\picture\images"       # 源图片目录
LABEL_DIR   = r"F:\RM-Demo\b\picture\labels"       # 源标注目录
OUTPUT_DIR  = r"F:\RM-Demo\b\picture"               # 输出根目录（生成 train/ val/ 子目录）
TRAIN_RATIO = 0.8                                   # 训练集比例
RANDOM_SEED = 42                                    # 随机种子（保证可复现）
# ====================

TRAIN_IMG   = Path(OUTPUT_DIR) / "train" / "images"
TRAIN_LBL   = Path(OUTPUT_DIR) / "train" / "labels"
VAL_IMG     = Path(OUTPUT_DIR) / "val"   / "images"
VAL_LBL     = Path(OUTPUT_DIR) / "val"   / "labels"


def split_and_copy():
    img_path   = Path(IMAGE_DIR)
    label_path = Path(LABEL_DIR)

    # 创建目标目录
    for d in (TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL):
        d.mkdir(parents=True, exist_ok=True)

    # 收集所有图片并排序
    img_files = sorted(
        [f for f in img_path.iterdir()
         if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda f: int(f.stem) if f.stem.isdigit() else f.stem
    )

    if not img_files:
        print(f"[提示] 未找到图片: {IMAGE_DIR}")
        return

    # 打乱并分割
    random.seed(RANDOM_SEED)
    random.shuffle(img_files)
    split_n = int(len(img_files) * TRAIN_RATIO)
    train_files = img_files[:split_n]
    val_files   = img_files[split_n:]

    print(f"总图片数：{len(img_files)}")
    print(f"训练集：{len(train_files)} 张  验证集：{len(val_files)} 张\n")

    def copy_pair(files, dst_img_dir, dst_lbl_dir, split_name):
        no_label = 0
        for img_f in files:
            lbl_f = label_path / f"{img_f.stem}.txt"
            shutil.copy2(img_f, dst_img_dir / img_f.name)
            if lbl_f.exists():
                shutil.copy2(lbl_f, dst_lbl_dir / lbl_f.name)
            else:
                (dst_lbl_dir / f"{img_f.stem}.txt").touch()
                no_label += 1
        print(f"  [{split_name}] {len(files)} 张图片，{no_label} 个空标注")

    copy_pair(train_files, TRAIN_IMG, TRAIN_LBL, "train")
    copy_pair(val_files,   VAL_IMG,   VAL_LBL,   "val")

    print(f"\n输出目录：")
    print(f"  {TRAIN_IMG}")
    print(f"  {TRAIN_LBL}")
    print(f"  {VAL_IMG}")
    print(f"  {VAL_LBL}")


if __name__ == "__main__":
    split_and_copy()
