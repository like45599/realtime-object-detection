# 实时目标检测系统（基于 YOLOv5）

本项目实现了一个基于 YOLOv5 的实时目标检测系统，能够从远程视频流中获取数据并进行目标检测。使用 PyTorch 和 OpenCV 进行实时视频流处理，能够对视频中的目标进行实时检测和展示。
CSDN笔记https://blog.csdn.net/like455/article/details/143686727

## 项目结构

```
realtime-object-detection/            # 项目根目录
├── assets/                           # 存放所有模型和相关资源
│   └── yolov5s.pt                    # 保存的 YOLOv5 模型权重
├── config/                           # 存放配置文件
│   └── config.py                     # 配置文件，存放视频 URL 和模型路径等
├── model/                            # 存放模型加载和推理相关代码
│   └── yolov5_model.py               # 加载和进行目标检测的代码
├── video/                            # 存放视频流相关功能
│   └── video_stream.py               # 获取视频流并传递给目标检测模型
├── main.py                           # 主程序，执行目标检测任务
└── requirements.txt                   # 记录项目依赖的库
```

## 安装与依赖

1. 克隆本项目到本地：
    ```bash
    git clone https://github.com/like45599/realtime-object-detection.git
    cd realtime-object-detection
    ```

2. 创建并激活虚拟环境（推荐使用 Anaconda 或 venv）：
    ```bash
    conda create -n realtime-object-detection python=3.9
    conda activate realtime-object-detection
    ```

3. 安装所需的 Python 库：
    ```bash
    pip install -r requirements.txt
    ```

    `requirements.txt` 文件的内容：
    ```
    torch==2.4.1
    opencv-python==4.7.0.72
    requests
    numpy
    ```

4. 如果没有预训练的 YOLOv5 模型，可以从 [YOLOv5 GitHub](https://github.com/ultralytics/yolov5) 下载并保存在 `assets/` 文件夹下，或者运行程序时自动下载。
   这边建议自动下载，经过热心网友**wangbowen-8800**的指正，本项目存在clone后无法直接运行、色彩失真的情况。
对应解决方案：
 - clone后无法直接运行：清空/assets 下的yolo5s.pt文件，以及项目目录下的yolo5s.pt直接自动下载。
 - 在video/video_stream.py代码中的`cv2.imshow("Video Stream with Detection", frame)`上一行加入`frame = frame[..., ::-1]`


## 配置

在 `config/config.py` 中设置你的视频流 URL 和模型文件路径。

示例：

```python
# config.py

VIDEO_URL = "http://xxx.xxx.xxx.xxx:5000/video_feed"  # 替换为 Flask 服务器的 URL
MODEL_PATH = './assets/yolov5s.pt'  # 模型文件存放路径
```

## 运行程序

1. 启动远程视频流（例如，使用 Flask 或其他服务器技术提供视频流）。

2. 运行主程序来启动实时目标检测：

    ```bash
    python main.py
    ```

3. 程序将从 `VIDEO_URL` 获取视频流，并将视频帧传递给 YOLOv5 模型进行目标检测。检测结果将在一个新的窗口中显示出来。

4. 按 `q` 键退出程序。

## 代码说明

### 1. `config/config.py`

`config.py` 用于存储项目中的配置信息，包括视频流的 URL 和模型的存储路径等。

示例代码：

```python
# 配置文件，存放 URL 和模型路径等
VIDEO_URL = "http://xxx.xxx.xxx.xxx:5000/video_feed"  # 替换为 Flask 服务器的 URL
MODEL_PATH = './assets/yolov5s.pt'  # 模型文件存放路径
```

### 2. `model/yolov5_model.py`

`yolov5_model.py` 文件包含了模型加载和目标检测的核心代码。

- **`load_model()`**：尝试从本地加载模型文件，如果本地文件不存在，则从 YOLOv5 官方仓库下载并保存到本地。
- **`detect_objects()`**：接收视频帧并使用 YOLOv5 进行目标检测，返回处理后的帧。

### 3. `video/video_stream.py`

`video_stream.py` 文件用于获取视频流，并将每一帧传递给目标检测函数进行处理。

```python
import cv2

def get_video_stream(url, detect_objects, model):
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 对每一帧进行目标检测
        frame = detect_objects(model, frame)

        # 显示检测结果
        cv2.imshow("Detection", frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```

### 4. `main.py`

`main.py` 是项目的入口文件，负责组织模型加载、视频流获取以及目标检测功能。

```python
import torch
from config.config import VIDEO_URL
from model.yolov5_model import load_model, detect_objects
from video.video_stream import get_video_stream

def main():
    # 加载 YOLOv5 模型
    model = load_model()

    # 获取远程视频流并进行目标检测
    get_video_stream(VIDEO_URL, detect_objects, model)

if __name__ == '__main__':
    main()
```

## 注意事项

- **视频流**：确保你的服务器正在提供有效的视频流。你可以使用 Flask 或其他技术来实现视频流服务。
- **YOLOv5 模型**：如果没有预训练的 YOLOv5 模型，程序会自动从 YOLOv5 官方仓库下载，并保存在 `assets/` 文件夹中。

## 常见问题

1. **如何更换模型？**
    - 你可以修改 `config.py` 中的 `MODEL_PATH` 变量，指定新的 YOLOv5 模型路径，或者下载其他预训练的 YOLOv5 模型（例如 `yolov5m.pt`、`yolov5l.pt` 等）。

2. **如何设置不同的网络摄像头 URL？**
    - 修改 `config.py` 中的 `VIDEO_URL` 变量，设置你的视频流 URL。

## 贡献

欢迎提交问题或 Pull Requests。如果你有任何建议或问题，可以在 GitHub 上创建 Issue 或与我联系。


