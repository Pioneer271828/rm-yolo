"""
数据集整理脚本：
1. 将 image 目录中的 PNG 图片转换为 JPG
2. 按文件名排序后，从指定编号开始重命名（32406, 32407, ...）
3. 标注文件同步重命名（无标注则生成空 txt）
4. 原 PNG 文件删除
"""

from pathlib import Path
from PIL import Image

# ===== 配置区域 =====
IMAGE_DIR  = r"F:\RM-Demo\b\picture\image"    # 图片目录（PNG）
LABEL_DIR  = r"F:\RM-Demo\b\picture\labels"   # 标注目录（TXT）
START_IDX  = 32406                             # 起始编号
# ====================


def process(image_dir: str, label_dir: str, start_idx: int):
    img_path   = Path(image_dir)
    label_path = Path(label_dir)

    # 收集并排序所有图片（支持 png/jpg/jpeg）
    img_files = sorted(
        [f for f in img_path.iterdir()
         if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg")],
        key=lambda f: f.name
    )

    if not img_files:
        print(f"[提示] 目录中未找到图片: {image_dir}")
        return

    print(f"共找到 {len(img_files)} 张图片，起始编号 {start_idx}，开始处理...\n")

    converted, renamed_label, empty_label = 0, 0, 0

    for i, src_img in enumerate(img_files):
        new_name   = str(start_idx + i)
        dst_jpg    = img_path  / f"{new_name}.jpg"
        dst_label  = label_path / f"{new_name}.txt"
        src_label  = label_path / f"{src_img.stem}.txt"

        # ── 图片转 JPG 并重命名 ──
        try:
            with Image.open(src_img) as im:
                im.convert("RGB").save(dst_jpg, "JPEG", quality=95)
            src_img.unlink()          # 删除原文件
            converted += 1
        except Exception as e:
            print(f"  [失败] {src_img.name}: {e}")
            continue

        # ── 标注文件同步重命名 ──
        if src_label.exists():
            src_label.rename(dst_label)
            renamed_label += 1
            label_tag = "标注已同步"
        else:
            dst_label.touch()         # 无标注 → 生成空 txt
            empty_label += 1
            label_tag = "无标注(空txt)"

        print(f"  [OK] {src_img.name}  →  {dst_jpg.name}   {label_tag}")

    print(f"\n处理完成：")
    print(f"  图片转换 & 重命名：{converted} 张")
    print(f"  标注同步重命名：  {renamed_label} 个")
    print(f"  生成空标注文件：  {empty_label} 个")
    print(f"  图片范围：{start_idx}.jpg ~ {start_idx + converted - 1}.jpg")


if __name__ == "__main__":
    process(IMAGE_DIR, LABEL_DIR, START_IDX)
