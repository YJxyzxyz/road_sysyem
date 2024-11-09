import cv2
import numpy as np

def calculate_image_contrast(image):
    # 转换图像为灰度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算图像的梯度
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅度
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # 计算对比度
    contrast = np.std(gradient_magnitude)

    return contrast

def calculate_image_color_change(image):
    # 转换图像为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 计算颜色变化
    color_change = np.std(hsv_image[:, :, 1])  # 使用HSV空间的饱和度通道

    return color_change

def calculate_image_brightness(image):
    # 转换图像为灰度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算图像亮度
    brightness = np.mean(gray_image)

    return brightness

def is_image_hazy(image_path, contrast_threshold=10, color_change_threshold=20, brightness_threshold=100):
    # 读取图像
    image = cv2.imread(image_path)

    # 计算图像对比度、颜色变化和亮度
    image_contrast = calculate_image_contrast(image)
    image_color_change = calculate_image_color_change(image)
    image_brightness = calculate_image_brightness(image)

    # 根据阈值判断是否有雾气
    if (image_contrast < contrast_threshold) and (image_color_change < color_change_threshold) and (image_brightness > brightness_threshold):
        return True
    else:
        return False

# 指定图像路径
image_path = './test/forest1.jpg'

# 设定阈值，可以根据实际情况调整
contrast_threshold_value = 10
color_change_threshold_value = 20
brightness_threshold_value = 100

# 判断图像是否带有雾气
result = is_image_hazy(image_path, contrast_threshold_value, color_change_threshold_value, brightness_threshold_value)

# 打印结果
if result:
    print("这张图片带有雾气。")
else:
    print("这张图片没有雾气。")
