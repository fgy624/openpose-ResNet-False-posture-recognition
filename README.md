# vr5.2
## 模型文件
model.h5等下载链接：
链接：https://pan.baidu.com/s/12Gt82egh2KIA_OTQycSdzg?pwd=aB12 
提取码：aB12 
### 1. model.h5
在下载好后，将model.h5放入到model/keras下
### 2. ResNet相关模型
就是所有名字开头带ResNet的模型文件，主要分为两类
**xxx_GPU.pth**是训练好的ResNet模型，根据其中的数字可分为ResNet34和ResNet101两种，没啥必要用
**xxx-乱码.pth**是预训练模型，可直接放在根目录下
## 代码执行
### 1. 安装包
首先直接运行`pip install -r requirement.txt `
之后到classify/ResNet目录下运行`pip install -r requirement.txt `
### 2. 运行
直接运行`python main.py即可`
