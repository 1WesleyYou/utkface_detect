import torch
import torch.nn as nn


class AgeGenderDetectNN(nn.Module):
    def __init__(self):
        super(AgeGenderDetectNN, self).__init__()
        self.age_detect_model = nn.Sequential(
            # 年纪检测
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 输入的是彩色图像
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.gender_detect_model = nn.Sequential(
            # 年纪检测
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 输入的是彩色图像
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.gender_fc1 = nn.Linear(128 * 64 * 64, 64)
        self.gender_fc2 = nn.Linear(64, 1)
        self.age_fc1 = nn.Linear(128 * 64 * 64, 64)
        self.age_fc2 = nn.Linear(64, 1)  # 做回归任务不用激活

    def forward(self, x):
        age = self.age_detect_model(x)
        age = age.view(-1, 128 * 64 * 64)
        age = self.age_fc1(age)
        # 年龄输出的应该是预测年龄
        age = self.age_fc2(age)

        gender = self.gender_detect_model(x)
        gender = gender.view(-1, 128 * 64 * 64)
        gender = self.gender_fc1(gender)
        gender = self.gender_fc2(gender)
        # gender 输出的应该是性别为 男0 的概率, 因此需要考虑使用激活函数，同时由于是二元分类，我们采用最简单的 sigmoid 函数就好
        gender = torch.sigmoid(gender)

        age = age.view(-1)
        gender = gender.view(-1)

        return age, gender
