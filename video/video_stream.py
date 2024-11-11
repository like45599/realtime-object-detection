import requests
import numpy as np
import cv2

# 获取远程视频流并进行目标检测
def get_video_stream(url, detect_function, model):
    # 请求视频流
    stream = requests.get(url, stream=True)

    # 初始化视频流读取
    bytes_data = b""
    for chunk in stream.iter_content(chunk_size=1024):
        bytes_data += chunk

        a = bytes_data.find(b'\xff\xd8')  # 查找JPEG图像开始
        b = bytes_data.find(b'\xff\xd9')  # 查找JPEG图像结束

        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]  # 提取单个图像
            bytes_data = bytes_data[b+2:]  # 清理已处理的字节

            # 转换为 OpenCV 格式
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            if frame is not None:
                # 调用目标检测函数
                frame = detect_function(model, frame)

                # 显示检测后的帧
                cv2.imshow("Video Stream with Detection", frame)

                # 调试：打印每一帧的尺寸
                print("Frame size:", frame.shape)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()