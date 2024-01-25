import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image

# 定义 transform 内容
transform = transforms.Compose([
    transforms.Resize([256, 256]),  # 重新设定图像大小
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5])
])


# 定义数据集获取类
class ImageDataset(Dataset):
    def __init__(self, transform_model=None):
        self.transform = transform_model
        self.data = self.load_data()

    def load_data(self):
        # 初始化空数据集
        data = []

        # 训练集读取
        for filename in os.listdir('dataset/train/part1'):
            age_r = filename.find('_')
            model_age = filename[:age_r]
            rest_name = filename[age_r + 1:]
            model_gender = int(rest_name[:rest_name.find('_')])
            data.append({'image_path': os.path.join(f'dataset/train/part1/{filename}'),
                         'label': (int(model_age), model_gender)})  # 获取 label

        # 训练集读取
        for filename in os.listdir('dataset/train/part2'):
            age_r = filename.find('_')
            model_age = filename[:age_r]
            rest_name = filename[age_r + 1:]
            model_gender = int(rest_name[:rest_name.find('_')])
            # print(f"age: {model_age}, gender: {model_gender}")
            data.append({'image_path': os.path.join(f'dataset/train/part2/{filename}'),
                         'label': (int(model_age), model_gender)})  # 获取 label 保存为一个元组

        # print(data[1000]['label'])
        return data

    def __len__(self):
        # 获取数据集的尺寸（返回文件个数）
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]['image_path']
        label = torch.tensor(self.data[idx]['label'])
        image = Image.open(img_path)  # .convert('L')

        if self.transform:
            image = self.transform(image)

        return {'image_path': image, 'label': label}


train_dataset = ImageDataset(transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 这里的 batch_size 是批次处理的数据量，根据电脑性能自己定
