from preload import train_loader, test_loader
from model import AgeGenderDetectNN
import torch
import torch.nn as nn
import torch.optim as optim

model = AgeGenderDetectNN()
# 损失函数定义
age_criterion = nn.L1Loss()  # 这里的使用的是MAE，或者说L1损失函数
gender_criterion = nn.BCELoss()  # 二元判断用二元交叉熵损失
# 优化器统一使用 Adam
age_optimizer = optim.Adam(model.parameters(), lr=0.001)
gender_optimizer = optim.Adam(model.parameters(), lr=0.001)

epoch_num = 10

for epochs in range(epoch_num):
    model.train()
    for idx, batch in enumerate(train_loader):
        input_image = batch['image_path']

        label = batch['label']
        label_age = label[:, 0]
        label_gender = label[:, 1]
        # 默认是 long，解释器不开心了
        label_age = label_age.float()
        label_gender = label_gender.float()

        age_optimizer.zero_grad()
        gender_optimizer.zero_grad()

        age_output, gender_output = model(input_image)
        age_loss = age_criterion(age_output, label_age)
        if max(gender_output) > 1:
            print(max(gender_output))
        if max(label_gender) > 1:
            print(max(label_gender))
        gender_loss = gender_criterion(gender_output, label_gender)

        age_loss.backward()
        gender_loss.backward()

        age_optimizer.step()
        gender_optimizer.step()
        print(idx)

        if idx % 100 == 1:
            print(f'case [{epochs}/21], loss of age = {age_loss.item()}, loss of gender = {gender_loss.item()}')

model.eval()
correct = 0
total = 0
with torch.no_grad():  # 关闭梯度计算，提高代码执行效率
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
