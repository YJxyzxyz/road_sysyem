import torch
import cv2
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
from tqdm import tqdm

# -----------------输入图像、深度图、雾图路径--------------#
img_path = Path(r'F:\frame\road')
depth_path = Path(r'F:\frame\road_depth')
hazy_path = Path(r'F:\frame\road_haze')
# -----------------雾气强度控制因子----------------------#
fog_strength = 1.0
# ----------------雾气颜色 (浅灰色雾气)-------------------#
fog_color = np.array([200, 200, 200], dtype=np.uint8)
# ------------------------------------------------------#
#   可选择的model: 'MiDas'、'MiDaS_small'、'DPT_Hybrid'
#   'MiDas'生成的深度图比MiDaS_small精度更高，适合一般的深度估计任务
#   'DPT_Hybrid'适合复杂场景下的深度估计
# ------------------------------------------------------#
model = 'DPT_Hybrid'

# ------------------------生成深度图---------------------#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载 MiDaS 模型
midas = torch.hub.load("intel-isl/MiDaS", model)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
midas.to(device)
midas.eval()

imglist = os.listdir(img_path)
with tqdm(total=len(imglist), desc=('深度图转换')) as pbar:
    for img in imglist:
        full_path = img_path / img
        image = cv2.imread(str(full_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if model == 'MiDas':
            transform = midas_transforms.default_transform
        elif model == 'DPT_Hybrid':
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform
        input = transform(image).to(device)

        with torch.no_grad():
            predict = midas(input)
        depth_map = predict.squeeze().cpu().numpy()
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        # # 可视化深度图
        # plt.imshow(depth_map_normalized, cmap='plasma')
        # plt.colorbar()
        # plt.title('Estimated Depth Map')
        # plt.show()

        depth_map = (depth_map_normalized * 255).astype(np.uint8)
        depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        # 保存深度图
        new_filename = depth_path / img
        new_filename = new_filename.with_suffix('.jpg')  # 可以自己更改深度图的格式，默认为png
        cv2.imwrite(str(new_filename), depth_map)

        pbar.update(1)

# -----------------------生成雾图----------------------------------#
with tqdm(total=len(imglist), desc=('雾图生成')) as pbar:
    for filename in os.listdir(img_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(img_path, filename)
            depthmap = os.path.join(depth_path, filename)

            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            depth_map = cv2.imread(depthmap, cv2.IMREAD_GRAYSCALE)

            if depth_map is None or original_image is None:
                print(f"跳过文件: {filename}，因为无法读取对应的深度图或原始图像。")
                print(
                    '请检查输入图像与深度图是否属于同一类型，如均为.png或.jpg，深度图的格式默认为.png，可自行更改其类型，详情请查看代码注释部分')
                continue

            depth_map_normalized = depth_map.astype(np.float32) / 255
            # 反转深度图，使得白色区域（近处）雾气浓，黑色区域（远处）雾气稀薄
            depth_map_inverted = 1 - depth_map_normalized
            # 雾气强度基于反转后的深度图，应用强度因子控制
            fog_intensity_map = depth_map_inverted * fog_strength
            # 限制雾气浓度的最大值为 1，避免过度曝光
            fog_intensity_map = np.clip(fog_intensity_map, 0, 1)
            fog_layer = np.ones_like(original_image, dtype=np.float32) * fog_color
            foggy_image = original_image * (1 - fog_intensity_map[:, :, np.newaxis]) + \
                          fog_layer * fog_intensity_map[:, :, np.newaxis]
            foggy_image = np.clip(foggy_image, 0, 255).astype(np.uint8)

            output_path = os.path.join(hazy_path, filename)
            foggy_image_bgr = cv2.cvtColor(foggy_image, cv2.COLOR_RGB2BGR)  # 转回 BGR 格式以保存
            cv2.imwrite(output_path, foggy_image_bgr)

        pbar.update(1)