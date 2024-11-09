import cv2
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from torch.autograd import Variable
import numpy as np

# 加载预训练的去雾模型
model_path = 'model_pretrained/AOD_net_epoch_relu_10.pth'  # 替换为您的模型路径
net = torch.load(model_path)
cuda_available = torch.cuda.is_available()

# 打开USB摄像头
usb_camera_index = 1  # 根据您的系统配置和USB摄像头索引更改
cap = cv2.VideoCapture(usb_camera_index)

# ===== Load input image =====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
]
)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        original_frame = frame.copy()  # 复制原始帧用于对比显示
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame).unsqueeze_(0)
        varIn = Variable(frame)
        if cuda_available:
            varIn = varIn.cuda()

        # 使用模型进行去雾
        dehazed_frame = net(varIn)

        # 将处理后的帧转换回NumPy数组，并调整像素值范围
        dehazed_frame = dehazed_frame.cpu().data.numpy().squeeze(0).transpose((1, 2, 0))
        dehazed_frame = (dehazed_frame * 255).astype(np.uint8)

        # 将颜色空间从RGB转换回BGR
        dehazed_frame = cv2.cvtColor(dehazed_frame, cv2.COLOR_RGB2BGR)

        # 显示原始帧
        cv2.imshow('Original Frame', original_frame)
        # 显示去雾后的帧
        cv2.imshow('Dehazed Frame', dehazed_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # 按下ESC键退出循环
            break
    else:
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()