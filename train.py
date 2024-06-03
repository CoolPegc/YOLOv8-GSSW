
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO("  F:\pjc\insulator\ultralytics\model\yolov8n.pt")

    # 训练模型
    results = model.train(data="F:\pjc\insulator\ultralytics\mydata\apple.yaml",
                          resume=True,
                          epochs=300,
                          batc=16,
                          project='apple',
                          patience=30,
                          name='yolov8n',
                          amp=False)

