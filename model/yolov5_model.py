import torch
from config.config import MODEL_PATH

# 加载 YOLOv5 模型
def load_model():
    # 创建 YOLOv5 模型（注意：这里只加载模型架构，权重会在后面加载）
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)  # 不加载预训练权重

    try:
        # 尝试加载本地模型权重
        model.load_state_dict(torch.load(MODEL_PATH))  # 从 assets 目录加载模型权重
        print(f"Loaded YOLOv5 model from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Model file not found at {MODEL_PATH}, downloading...")
        # 如果模型权重不存在，从官网加载
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 加载预训练的 yolov5s 模型
        # 保存模型权重到 assets 目录
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model downloaded and saved to {MODEL_PATH}")

    model.eval()  # 设置模型为评估模式
    return model

# 目标检测函数
def detect_objects(model, frame):
    # 将 BGR 图像转换为 RGB
    img = frame[..., ::-1]

    # 使用 YOLOv5 进行目标检测
    results = model(img)

    # 绘制检测结果
    results.render()

    # 获取处理后的图像，使用 results.ims 来代替 results.imgs
    frame = results.ims[0]

    return frame