"""
将指定目录中的 JPG/JPEG 图片批量转换为 PNG 格式
"""

import os
from pathlib import Path
from PIL import Image


def convert_jpg_to_png(input_dir: str, output_dir: str = None, delete_original: bool = False):
    """
    将目录中的 JPG/JPEG 图片转换为 PNG

    :param input_dir: 输入目录路径
    :param output_dir: 输出目录路径，为 None 则保存到同一目录
    :param delete_original: 转换成功后是否删除原始 JPG 文件
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"[错误] 目录不存在: {input_dir}")
        return

    # 若未指定输出目录，输出到输入目录
    out_path = Path(output_dir) if output_dir else input_path
    out_path.mkdir(parents=True, exist_ok=True)

    # 收集所有 jpg/jpeg 文件（不区分大小写）
    jpg_files = [f for f in input_path.iterdir()
                 if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg")]

    if not jpg_files:
        print(f"[提示] 目录中未找到 JPG/JPEG 文件: {input_dir}")
        return

    print(f"共找到 {len(jpg_files)} 张 JPG 图片，开始转换...\n")
    success, failed = 0, 0

    for jpg_file in jpg_files:
        png_file = out_path / (jpg_file.stem + ".png")
        try:
            with Image.open(jpg_file) as img:
                # 转为 RGB 以避免透明通道问题
                img.convert("RGB").save(png_file, "PNG")
            print(f"  [OK] {jpg_file.name}  →  {png_file.name}")
            success += 1
            if delete_original:
                jpg_file.unlink()
        except Exception as e:
            print(f"  [失败] {jpg_file.name}: {e}")
            failed += 1

    print(f"\n转换完成：成功 {success} 张，失败 {failed} 张")
    print(f"PNG 文件保存至: {out_path.resolve()}")


if __name__ == "__main__":
    # ===== 配置区域 =====
    INPUT_DIR = r"F:\RM-Demo\b\picture\picture"  # JPG 图片所在目录
    OUTPUT_DIR = None          # 输出目录，None 表示与输入目录相同
    DELETE_ORIGINAL = True     # 转换后是否删除原始 JPG 文件
    # ====================

    convert_jpg_to_png(INPUT_DIR, OUTPUT_DIR, DELETE_ORIGINAL)
