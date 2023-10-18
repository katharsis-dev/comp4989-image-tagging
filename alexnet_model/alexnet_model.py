# Original code located at: https://github.com/madsendennis/notebooks/tree/master/pytorch

from torchvision import models
from torchvision import transforms
from PIL import Image
from skimage import io, transform

import torch
import matplotlib.pyplot as plt

def main():
    alexnet = models.alexnet(pretrained=True)

    img = Image.open('test_img_4.jpg')

    # resize/process image
    size = 255
    img.resize((256, 256))

    # resize image to 256x256, crop it to 224x224, convert the image to 
    # PyTorch Tensor data type, normalize it and
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    
    img_train = transform(img)
    
    batch_train = torch.unsqueeze(img_train, 0)

    alexnet.eval()

    out = alexnet(batch_train)

    with open('alexnet_model/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    results = [(classes[idx], percentage[idx].item()) for idx in indices[0][:10]]
    for r in results:
        print(r)



if __name__ == "__main__":
    main()