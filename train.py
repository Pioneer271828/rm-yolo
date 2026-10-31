"""
YOLOv8 装甲板识别训练脚本
训练模型识别RM比赛中的装甲板（14个类别）
"""

import os
from ultralytics import YOLO


def train_armor_detection():
    """
    训练装甲板检测模型
    """
    
    # ==================== 配置参数 ====================
    
    # 数据集配置文件路径
    dataset_yaml = 'dataset.yaml'
    
    # 模型配置
    model_type = 'yolov8l.pt'  # 可选: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
                                # n: nano (最快), s: small, m: medium, l: large, x: xlarge (最准)
    
    # 检查模型文件是否存在，不存在则自动从官方下载
    if not os.path.exists(model_type):
        print(f"本地未找到 {model_type}，正在自动下载...")
        model = YOLO(model_type)  # Ultralytics 会自动下载到本地缓存并加载
        import shutil, ultralytics
        cache_path = os.path.join(os.path.dirname(ultralytics.__file__), 'assets', model_type)
        hub_path = os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'Ultralytics', model_type)
        for src in [cache_path, hub_path]:
            if os.path.exists(src):
                shutil.copy(src, model_type)
                print(f"已将 {model_type} 保存到项目目录")
                break
        else:
            print(f"{model_type} 已下载，由 Ultralytics 缓存管理")
    else:
        print(f"检测到本地模型文件: {model_type}")
        model = YOLO(model_type)
    
# 训练参数
    epochs = 150              # 训练轮数
    batch_size = 8          # 批次大小（根据显存调整）
    imgsz = [480, 768]        # 输入图像尺寸 [height, width]，YOLOv8格式为[高, 宽]
    device = '0'              # 设备: '0' (GPU 0), '1' (GPU 1), 'cpu', '0,1' (多GPU)
    
    # 保存设置 - 每10轮保存一次
    save_period = 10          # 每隔多少轮保存一次模型
    project = 'runs/train'    # 训练结果保存目录
    name = 'armor_detection'  # 实验名称
    
    # 其他优化参数
    lr0 = 0.00001                # 初始学习率
    lrf = 0.00001                # 最终学习率 (lr0 * lrf)
    momentum = 0.937          # SGD动量
    weight_decay = 0.0005     # 权重衰减
    warmup_epochs = 3         # 预热轮数
    warmup_momentum = 0.8     # 预热动量
    warmup_bias_lr = 0.1      # 预热偏置学习率
    
    # 数据增强参数
    hsv_h = 0.015             # 色调增强
    hsv_s = 0.7               # 饱和度增强
    hsv_v = 0.4               # 明度增强
    degrees = 0.0             # 旋转角度（装甲板一般不需要旋转）
    translate = 0.1           # 平移
    scale = 0.5               # 缩放
    shear = 0.0               # 剪切
    perspective = 0.0         # 透视变换
    flipud = 0.0              # 上下翻转
    fliplr = 0.5              # 左右翻转
    mosaic = 0.8              # 马赛克增强概率（0.8表示80%概率使用mosaic）
    mixup = 0.0               # Mixup增强
    # 注意：平行动态模糊需要通过自定义数据增强实现
    # YOLOv8原生不支持motion_blur参数，可通过 albumentations 库实现
    # 注意：YOLOv8原生不支持mosaic缩放范围参数，默认缩放范围通过scale参数控制
    
    # 验证参数
    val = True                # 训练过程中进行验证
    plots = True              # 保存训练曲线图
    
    # ==================== 开始训练 ====================
    
    print("=" * 60)
    print("开始训练装甲板检测模型")
    print("=" * 60)
    print(f"数据集配置: {dataset_yaml}")
    print(f"模型类型: {model_type}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"图像尺寸: {imgsz}")
    print(f"保存间隔: 每 {save_period} 轮保存一次")
    print(f"设备: {device}")
    print("=" * 60)
    
    # 训练模型
    print("\n开始训练...\n")
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        
        # 保存设置
        save_period=save_period,
        project=project,
        name=name,
        
        # 优化参数
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        warmup_momentum=warmup_momentum,
        warmup_bias_lr=warmup_bias_lr,
        
        # 数据增强
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        flipud=flipud,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        
        # 验证
        val=val,
        plots=plots,
        
        # 其他设置
        exist_ok=True,           # 允许覆盖已存在的文件夹
        pretrained=True,         # 使用预训练权重
        verbose=True,            # 显示详细训练信息
        seed=42,                 # 随机种子（保证可重复性）
        deterministic=True,      # 使用确定性算法
        single_cls=False,        # 多类别分类
    )
    
    # ==================== 训练完成 ====================
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"\n最佳模型保存在: {results.save_dir}/weights/")
    print(f"最佳模型文件: best.pt")
    print(f"最后模型文件: last.pt")
    print(f"\n所有保存的检查点: 每{save_period}轮一次")
    print(f"\n训练结果保存在: {results.save_dir}/")
    print(f"可视化图表: {results.save_dir}/results.png")
    print(f"混淆矩阵: {results.save_dir}/confusion_matrix.png")
    print(f"PR曲线: {results.save_dir}/PR_curve.png")
    print(f"F1曲线: {results.save_dir}/F1_curve.png")
    
    # 验证最佳模型
    print("\n" + "=" * 60)
    print("在验证集上评估最佳模型...")
    print("=" * 60)
    metrics = model.val(data=dataset_yaml, device=device)
    
    print("\n" + "=" * 60)
    print("最终评估结果:")
    print("=" * 60)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    return results, metrics


if __name__ == '__main__':
    # 训练模型
    results, metrics = train_armor_detection()
    
    print("\n" + "=" * 60)
    print("训练脚本执行完毕！")
    print("=" * 60)
