from transformers import AutoImageProcessor, ResNetForImageClassification
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
import pandas as pd
import torch

from model import MultiLabelClassifier
from dataset import MoviePosterDataset

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

data = pd.read_csv("./datasets/Movies-Poster_Dataset-master/train.csv")
data = data.head(1000)

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
cnn_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
model_config = {
    'model': cnn_model,
    'p_drop': 0.3,
    'n_model_out': 1000,
    'n_classes': 25,
}

image_paths = [''.join(["./datasets/Movies-Poster_Dataset-master/Images/", item, ".jpg"]) for item in data['Id']]
labels = data.drop(columns=["Id", "Genre"]).values.tolist()

dataset = MoviePosterDataset(image_paths, labels, processor)
model = MultiLabelClassifier(**model_config).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)


def run():
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        print(f"Epoch {epoch + 1}: ")

        for batch in train_loader:
            imgs = batch['image'].to(device)
            labs = batch['label'].to(device)

            optimizer.zero_grad()
            loss, _ = model(imgs, labs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f'Training loss: {train_loss / len(train_loader)}')

        model.eval()
        with torch.no_grad():
            eval_loss = 0
            precision = 0
            for batch in eval_loader:
                imgs = batch['image'].to(device)
                labs = batch['label'].to(device)

                loss, MAP = model(imgs, labs)
                eval_loss += loss.item()
                precision += MAP.item() * imgs.shape[0] / eval_size

            print(f"Mean average precision: {precision}")
            print(f"Evaluation loss: {eval_loss / len(eval_loader)}")


run()
