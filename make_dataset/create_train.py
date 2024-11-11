import os
import argparse
import numpy as np
from random import uniform
import h5py
from PIL import Image


def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255 * np.ones((len(arr), 1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)


parser = argparse.ArgumentParser()
parser.add_argument('--nyu', type=str, required=True, help='path to nyu_depth_v2_labeled.mat')
parser.add_argument('--dataset', type=str, required=True, help='path to  synthesized hazy images dataset store')
args = parser.parse_args()
print(args)

nyu_depth = h5py.File(args.nyu + '/nyu_depth_v2_labeled.mat', 'r')

directory = os.path.join(args.dataset, 'train')
saveimgdir = os.path.join(args.dataset, 'demo')
os.makedirs(directory, exist_ok=True)
os.makedirs(saveimgdir, exist_ok=True)

image = nyu_depth['images']
depth = nyu_depth['depths']

total_num = 0
for index in range(1445):
    gt_image = (image[index, :, :, :]).astype(float)
    gt_image = np.swapaxes(gt_image, 0, 2)

    # 使用PIL进行图像大小调整
    gt_image = Image.fromarray(gt_image.astype('uint8'))
    gt_image = gt_image.resize((640, 480), Image.LANCZOS)
    gt_image = np.array(gt_image).astype(float) / 255

    gt_depth = depth[index, :, :]
    maxhazy = gt_depth.max()
    gt_depth = gt_depth / maxhazy
    gt_depth = np.swapaxes(gt_depth, 0, 1)

    for j in range(7):
        for k in range(3):
            # beta
            bias = 0.05
            temp_beta = 0.4 + 0.2 * j
            beta = uniform(temp_beta - bias, temp_beta + bias)

            tx1 = np.exp(-beta * gt_depth)

            # A
            abias = 0.1
            temp_a = 0.5 + 0.2 * k
            a = uniform(temp_a - abias, temp_a + abias)
            A = [a, a, a]

            m, n = gt_image.shape[0], gt_image.shape[1]

            rep_atmosphere = np.tile(np.reshape(A, [1, 1, 3]), [m, n, 1])
            tx1 = np.reshape(tx1, [m, n, 1])

            max_transmission = np.tile(tx1, [1, 1, 3])

            haze_image = gt_image * max_transmission + rep_atmosphere * (1 - max_transmission)

            total_num += 1
            Image.fromarray((haze_image * 255).astype('uint8')).save(os.path.join(saveimgdir, f'haze_{total_num}.jpg'))
            Image.fromarray((gt_image * 255).astype('uint8')).save(os.path.join(saveimgdir, f'gt_{total_num}.jpg'))

            h5f = h5py.File(os.path.join(directory, f'{total_num}.h5'), 'w')
            h5f.create_dataset('haze', data=haze_image)
            h5f.create_dataset('gt', data=gt_image)
