"""
预测
"""
# -*- coding: utf-8 -*-
import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from classfy.ResNet.model import resnet34, resnet101


def main(input_img):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_path = input_img
    img = Image.open(image_path)
    # plt.imshow(img)
    img = data_transform(img)   # [N, C H, W]
    img = torch.unsqueeze(img, dim=0)   # 维度扩展
    # print(f"img={img}")
    json_path = "./classfy/ResNet/calss_indices.json"
    with open(json_path, 'r') as f:
        class_indict = json.load(f)

    # model = AlexNet(num_classes=5).to(device)   # GPU
    # model = vgg(model_name="vgg16", num_classes=5)  # CPU

    # model = resnet34(num_classes=12)
    model = resnet101(num_classes=12)
    weights_path = "./classfy/ResNet/ResNet101_GPU.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    class_name = ''
    with torch.no_grad():
        # output = torch.squeeze(model(img.to(device))).cpu()   #GPU
        output = torch.squeeze(model(img))      # 维度压缩
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        print_res = "class: {}  prob: {:.3}".format(class_indict[str(predict_cla)],
                                                    predict[predict_cla].numpy())
        
        class_name = class_indict[str(predict_cla)]
        plt.title(print_res)
        # for i in range(len(predict)):
        #     print("class: {}  prob: {:.3}".format(class_indict[str(predict_cla)],
        #                                             predict[predict_cla].numpy()))
        # plt.show()
        plt.savefig('./result.png')
    
    return class_name
# if __name__ == '__main__':
#     main()
