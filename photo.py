import os


def batch_rename_images(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 过滤出图片文件（假设图片格式是jpg或png，如果有其他格式可以补充）
    image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]

    for index, filename in enumerate(image_files):
        # 构造新文件名
        new_filename = f'haze_{index + 1}.jpg'  # 如果需要保留原始扩展名，请使用filename.split('.')[-1]

        # 获取旧文件的完整路径
        old_filepath = os.path.join(folder_path, filename)

        # 获取新文件的完整路径
        new_filepath = os.path.join(folder_path, new_filename)

        # 重命名文件
        os.rename(old_filepath, new_filepath)
        print(f'Renamed {old_filepath} to {new_filepath}')


# 指定包含图片的文件夹路径
folder_path = 'F:/frame/road_haze'

# 执行批量重命名
batch_rename_images(folder_path)
