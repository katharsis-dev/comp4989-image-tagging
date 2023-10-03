from transformers import AutoImageProcessor, ResNetForImageClassification
from datasets import load_dataset
import torch

from model import MultiLabelClassifier

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
cnn_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
classifier = MultiLabelClassifier()

inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    logits = cnn_model(**inputs).logits

pred = classifier(logits)

print(pred)



