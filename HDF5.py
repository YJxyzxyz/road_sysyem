import os
import h5py
import numpy as np
from PIL import Image


def save_images_to_h5(demo_dir, output_dir):
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取demo目录中的所有文件
    haze_images = sorted(f for f in os.listdir(demo_dir) if f.startswith('haze_') and f.endswith('.jpg'))
    gt_images = sorted(f for f in os.listdir(demo_dir) if f.startswith('gt_') and f.endswith('.jpg'))

    # 确保雾图和原图数量相同
    assert len(haze_images) == len(gt_images), "雾图和原图数量不匹配！"

    total_num = 0
    for haze_filename, gt_filename in zip(haze_images, gt_images):
        # 读取图像
        haze_image_path = os.path.join(demo_dir, haze_filename)
        gt_image_path = os.path.join(demo_dir, gt_filename)

        haze_image = np.array(Image.open(haze_image_path)).astype(float) / 255
        gt_image = np.array(Image.open(gt_image_path)).astype(float) / 255

        # 创建 H5 文件
        total_num += 1
        h5_file_path = os.path.join(output_dir, f'{total_num}.h5')
        with h5py.File(h5_file_path, 'w') as h5f:
            h5f.create_dataset('haze', data=haze_image)
            h5f.create_dataset('gt', data=gt_image)
        print(f'Saved {h5_file_path}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', type=str, required=True,
                        help='path to demo images directory containing haze and gt images')
    parser.add_argument('--output', type=str, required=True, help='path to store hdf5 dataset')
    args = parser.parse_args()

    save_images_to_h5(args.demo, args.output)
