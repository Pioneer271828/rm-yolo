"""
YOLOv8 装甲板识别推理/预测脚本
用于在图像、视频或摄像头流中检测装甲板
"""

import cv2
import os
from pathlib import Path
from ultralytics import YOLO


class ArmorDetector:
    """
    装甲板检测器类
    """
    
    # 类别名称映射
    CLASS_NAMES = {
        0: 'B1',   # 蓝方1号装甲板
        1: 'B2',   # 蓝方2号装甲板
        2: 'B3',   # 蓝方3号装甲板
        3: 'B4',   # 蓝方4号装甲板
        4: 'B5',   # 蓝方5号装甲板
        5: 'BO',   # 蓝方前哨站
        6: 'BS',   # 蓝方基地
        7: 'R1',   # 红方1号装甲板
        8: 'R2',   # 红方2号装甲板
        9: 'R3',   # 红方3号装甲板
        10: 'R4',  # 红方4号装甲板
        11: 'R5',  # 红方5号装甲板
        12: 'RO',  # 红方前哨站
        13: 'RS',  # 红方基地
    }
    
    # 颜色映射 (BGR格式)
    COLORS = {
        0: (255, 0, 0),     # B1 - 蓝色
        1: (255, 0, 0),     # B2 - 蓝色
        2: (255, 0, 0),     # B3 - 蓝色
        3: (255, 0, 0),     # B4 - 蓝色
        4: (255, 0, 0),     # B5 - 蓝色
        5: (255, 100, 0),   # BO - 蓝橙色
        6: (0, 0, 255),     # BS - 深蓝色
        7: (0, 0, 255),     # R1 - 红色
        8: (0, 0, 255),     # R2 - 红色
        9: (0, 0, 255),     # R3 - 红色
        10: (0, 0, 255),    # R4 - 红色
        11: (0, 0, 255),    # R5 - 红色
        12: (0, 100, 255),  # RO - 红橙色
        13: (0, 0, 139),    # RS - 深红色
    }
    
    def __init__(self, model_path=r'C:\Users\remote\Desktop\yolo\runs\detect\runs\train\armor_detection\weights\best.pt', device='0'):
        """
        初始化检测器
        
        Args:
            model_path: 模型权重文件路径
            device: 设备 ('0', 'cpu', etc.)
        """
        print(f"正在加载模型: {model_path}")
        self.model = YOLO(model_path)
        self.device = device
        print(f"模型加载成功！")
    
    def predict_image(self, image_path, conf_threshold=0.5, save_result=True, output_dir='runs/predict'):
        """
        在单张图像上进行装甲板检测
        
        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值
            save_result: 是否保存结果
            output_dir: 输出目录
            
        Returns:
            results: 检测结果
        """
        print(f"\n正在检测图像: {image_path}")
        
        # 进行预测
        results = self.model.predict(
            image_path,
            conf=conf_threshold,
            device=self.device,
            save=save_result,
            project=output_dir,
            name='images',
            exist_ok=True
        )
        
        # 打印检测结果
        self._print_results(results[0])
        
        return results
    
    def predict_video(self, video_path, conf_threshold=0.5, save_result=True, output_dir='runs/predict'):
        """
        在视频上进行装甲板检测
        
        Args:
            video_path: 视频路径
            conf_threshold: 置信度阈值
            save_result: 是否保存结果
            output_dir: 输出目录
            
        Returns:
            results: 检测结果
        """
        print(f"\n正在检测视频: {video_path}")
        
        # 进行预测
        results = self.model.predict(
            video_path,
            conf=conf_threshold,
            device=self.device,
            save=save_result,
            project=output_dir,
            name='videos',
            exist_ok=True
        )
        
        return results
    
    def predict_webcam(self, conf_threshold=0.5, camera_id=0):
        """
        使用摄像头进行实时装甲板检测
        
        Args:
            conf_threshold: 置信度阈值
            camera_id: 摄像头ID
        """
        print(f"\n正在启动摄像头 (ID: {camera_id})...")
        print("按 'q' 键退出")
        
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("无法打开摄像头！")
            return
        
        while True:
            # 读取帧
            ret, frame = cap.read()
            
            if not ret:
                print("无法读取摄像头画面！")
                break
            
            # 进行预测
            results = self.model.predict(
                frame,
                conf=conf_threshold,
                device=self.device,
                verbose=False
            )
            
            # 绘制检测结果
            annotated_frame = self._draw_results(frame, results[0])
            
            # 显示结果
            cv2.imshow('Armor Detection - Webcam', annotated_frame)
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("\n摄像头检测已停止")
    
    def predict_folder(self, folder_path, conf_threshold=0.5, save_result=True, output_dir='runs/predict'):
        """
        批量检测文件夹中的所有图像
        
        Args:
            folder_path: 文件夹路径
            conf_threshold: 置信度阈值
            save_result: 是否保存结果
            output_dir: 输出目录
        """
        print(f"\n正在批量检测文件夹: {folder_path}")
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # 获取所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(folder_path).glob(f'*{ext}'))
            image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"未在 {folder_path} 中找到图像文件！")
            return
        
        print(f"找到 {len(image_files)} 张图像")
        
        # 批量预测
        results = self.model.predict(
            str(folder_path),
            conf=conf_threshold,
            device=self.device,
            save=save_result,
            project=output_dir,
            name='batch',
            exist_ok=True
        )
        
        print(f"\n批量检测完成！结果保存在: {output_dir}/batch/")
        
        return results
    
    def _print_results(self, result):
        """
        打印检测结果
        
        Args:
            result: YOLO检测结果
        """
        boxes = result.boxes
        
        if len(boxes) == 0:
            print("未检测到装甲板")
            return
        
        print(f"检测到 {len(boxes)} 个装甲板:")
        print("-" * 60)
        
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            class_name = self.CLASS_NAMES.get(class_id, f'Unknown')
            confidence = float(box.conf[0])
            
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            print(f"  [{i+1}] 类别: {class_name} (ID: {class_id})")
            print(f"      置信度: {confidence:.4f}")
            print(f"      位置: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")
            print(f"      尺寸: {x2-x1:.1f} x {y2-y1:.1f}")
            print("-" * 60)
    
    def _draw_results(self, frame, result):
        """
        在图像上绘制检测结果
        
        Args:
            frame: 原始图像
            result: YOLO检测结果
            
        Returns:
            annotated_frame: 标注后的图像
        """
        boxes = result.boxes
        
        if len(boxes) == 0:
            return frame
        
        annotated_frame = frame.copy()
        
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = self.CLASS_NAMES.get(class_id, f'Unknown')
            confidence = float(box.conf[0])
            
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 获取颜色
            color = self.COLORS.get(class_id, (0, 255, 0))
            
            # 绘制边界框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f'{class_name} {confidence:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # 标签背景
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # 标签文字
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return annotated_frame


def main():
    """
    主函数 - 演示如何使用装甲板检测器
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='装甲板检测推理脚本')
    parser.add_argument('--model', type=str, default=r'C:\Users\remote\Desktop\yolo\runs\detect\runs\train\armor_detection\weights\epoch90.pt',
                        help='模型权重文件路径')
    parser.add_argument('--source', type=str,
                        help='输入源 (图像路径、视频路径、文件夹路径或摄像头ID)', default='WIN_20260305_20_38_55_Pro.mp4')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='置信度阈值')
    parser.add_argument('--device', type=str, default='0',
                        help='设备 (0: GPU 0, cpu: CPU)')
    parser.add_argument('--save', action='store_true', default=True,
                        help='保存检测结果')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = ArmorDetector(model_path=args.model, device=args.device)
    
    # 判断输入类型
    source = args.source.lower()
    
    if source.isdigit() and len(source) <= 2:
        # 摄像头
        camera_id = int(source)
        detector.predict_webcam(conf_threshold=args.conf, camera_id=camera_id)
    elif os.path.isfile(args.source):
        # 文件
        file_ext = Path(args.source).suffix.lower()
        if file_ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}:
            # 图像
            detector.predict_image(args.source, conf_threshold=args.conf, save_result=args.save)
        elif file_ext in {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}:
            # 视频
            detector.predict_video(args.source, conf_threshold=args.conf, save_result=args.save)
        else:
            print(f"不支持的文件格式: {file_ext}")
    elif os.path.isdir(args.source):
        # 文件夹
        detector.predict_folder(args.source, conf_threshold=args.conf, save_result=args.save)
    else:
        print(f"无效的输入源: {args.source}")


if __name__ == '__main__':
    main()