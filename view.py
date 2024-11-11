import h5py
import matplotlib.pyplot as plt

# 指定 HDF5 文件路径
file_path = './trainset/train/1.h5'

# 打开 HDF5 文件
with h5py.File(file_path, 'r') as f:
    # 读取数据集
    dataset = f['haze']
    data = dataset[:]

    # 将数据可视化
    plt.imshow(data)
    plt.show()
