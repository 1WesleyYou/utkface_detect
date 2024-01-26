# 面部年龄和性别检测

## 任务特点

我们首先要在训练的时候同时训练两个独立特征，这就需要 tuple 的思路了

同时我们要区分年龄的回归思路和性别的分类思路，回归是用于拟合函数，通过loss(MAE)函数获得损失然后优化系数；分类是离散式的，用神网输出男性的概率p，
这里从处理方便角度考虑需要用sigmoid函数进行优化，同样进行输出之后求loss和优化

## 数据集的问题

- 数据集中part2有个照片名称多打了一个 `_`即`53__0_20170116184028385.jpg`, 请注意消除之
- 数据集的性别有3出现，为 `part1/61_3_20170109150557335.jpg`

## 数据集优化

为了满足让模型能够识别更加抽象的人脸，我们使用 `transforms` 的内置功能进行数据集预处理，具体函数如下

```python
import torchvision.transforms as transforms
transforms.Compose([ 
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(degrees=15),  # 随机旋转图像
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机调整亮度和对比度
    transforms.RandomResizedCrop(size=(256, 256)),  # 随机裁剪和缩放
    transforms.ToTensor()
 ])

```