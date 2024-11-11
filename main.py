import cv2
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