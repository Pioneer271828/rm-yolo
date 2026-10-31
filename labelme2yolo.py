"""
将 LabelMe JSON 标注文件转换为 YOLO 检测格式 (.txt)
YOLO 格式：<class_id> <cx> <cy> <w> <h>  (均归一化到 [0,1])
"""

import json
import os
from pathlib import Path

# ===== 配置区域 =====
LABEL_DIR  = r"F:\RM-Demo\b\picture\label"   # LabelMe JSON 目录
OUTPUT_DIR = r"F:\RM-Demo\b\picture\labels"  # 输出 YOLO txt 目录

# 类别映射：标注中的 label → YOLO class_id
# 若 JSON 中出现未定义的 label，会打印警告并跳过
CLASS_MAP = {
    "enermy": 0,
    # 如有更多类别，按下面格式继续添加：
    # "B1": 1,
}
# ====================


def rect_to_yolo(points, img_w, img_h):
    """
    LabelMe 矩形两点 → YOLO 归一化中心框
    points: [[x1,y1], [x2,y2]]
    """
    x1, y1 = points[0]
    x2, y2 = points[1]
    # 兼容点顺序不固定的情况
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)

    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w  = (xmax - xmin) / img_w
    h  = (ymax - ymin) / img_h

    # 裁剪到 [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))
    return cx, cy, w, h


def convert(label_dir: str, output_dir: str, class_map: dict):
    label_path  = Path(label_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = sorted(label_path.glob("*.json"))
    if not json_files:
        print(f"[提示] 目录中未找到 JSON 文件: {label_dir}")
        return

    print(f"共找到 {len(json_files)} 个标注文件，开始转换...\n")
    success, skipped, warn_labels = 0, 0, set()

    for jf in json_files:
        with open(jf, encoding="utf-8") as fp:
            data = json.load(fp)

        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")

        if not img_w or not img_h:
            print(f"  [跳过] {jf.name}：缺少 imageWidth/imageHeight")
            skipped += 1
            continue

        lines = []
        for shape in data.get("shapes", []):
            label      = shape.get("label", "")
            shape_type = shape.get("shape_type", "")
            points     = shape.get("points", [])

            if label not in class_map:
                warn_labels.add(label)
                continue

            if shape_type == "rectangle" and len(points) == 2:
                cx, cy, w, h = rect_to_yolo(points, img_w, img_h)
                lines.append(f"{class_map[label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            else:
                print(f"  [警告] {jf.name}：不支持的 shape_type='{shape_type}'，已跳过")

        txt_file = output_path / (jf.stem + ".txt")
        with open(txt_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(lines))

        print(f"  [OK] {jf.name}  →  {txt_file.name}  ({len(lines)} 条)")
        success += 1

    print(f"\n转换完成：成功 {success} 个，跳过 {skipped} 个")
    print(f"YOLO 标注保存至: {output_path.resolve()}")

    if warn_labels:
        print(f"\n[警告] 以下标签未在 CLASS_MAP 中定义，已忽略: {warn_labels}")
        print("       请在脚本 CLASS_MAP 中添加对应映射。")


if __name__ == "__main__":
    convert(LABEL_DIR, OUTPUT_DIR, CLASS_MAP)
