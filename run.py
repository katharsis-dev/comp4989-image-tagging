from transformers import AutoImageProcessor, ResNetForImageClassification
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
import pandas as pd
import torch
import torchmetrics as tm
from tqdm import tqdm
import os
import time
import argparse
import random

from model import MultiLabelClassifier
from dataset import ImageDataset

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()
parser.add_argument('mode', help="Choose train, test, or train_debug.")
parser.add_argument('-d', '--dataset', help="Choose flickr or movie.")
args = parser.parse_args()

save_dir = 'models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if args.dataset == "movie":
    data = pd.read_csv("./datasets/Movies-Poster_Dataset-master/train.csv")
    data = data.head(100)
    image_paths = [''.join(["./datasets/Movies-Poster_Dataset-master/Images/", str(item), ".jpg"]) for item in data['Id']]
    labels = data.drop(columns=["Id", "Genre"]).values.tolist()
    n_classes = 25
else:
    data = pd.read_csv("./datasets/mirflickr25k/output.csv")
    data = data.head(100)
    image_paths = [''.join(["./datasets/mirflickr25k/mirflickr/im", str(item), ".jpg"]) for item in data['Image']]
    labels = data.drop(columns=["Image"]).values.tolist()
    n_classes = 24

if args.mode == 'train_debug':
    device = 'cpu'

random.seed(226)
torch.manual_seed(226)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(226)

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
cnn_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
model_config = {
    'model': cnn_model,
    'p_drop': 0.5,
    'n_model_out': 1000,
    'n_classes': n_classes,
}

dataset = ImageDataset(image_paths, labels, processor)
model = MultiLabelClassifier(**model_config).to(device)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
metric = tm.AveragePrecision('multilabel', num_labels=model_config["n_classes"])

train_size = int(0.7 * len(dataset))
eval_size = int(0.25 * len(dataset))
test_size = len(dataset) - train_size - eval_size
train_dataset, eval_dataset, test_dataset = random_split(dataset, [train_size, eval_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

save_file_name = f"model_state{'_movie' if args.dataset == 'movie' else ''}.pt"

def train():
    start = time.time()
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_pred = []
        train_labels = []

        print(f"Epoch {epoch + 1}: ")

        with tqdm(total=len(train_loader)) as prog:
            for batch in train_loader:
                imgs = batch['image'].to(device)
                labs = batch['label'].to(device)

                optimizer.zero_grad()
                loss, pred = model(imgs, labs)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * imgs.shape[0] / train_size
                train_pred.append(pred)
                train_labels.append(labs)

                prog.update(1)

        train_pred = torch.cat(train_pred)
        train_labels = torch.cat(train_labels)
        train_map = metric(train_pred, train_labels)
        print(f'Training loss: {train_loss}')
        print(f"Mean average precision: {train_map}")

        model.eval()
        with torch.no_grad():
            eval_loss = 0
            all_pred = []
            all_labels = []

            for batch in eval_loader:
                imgs = batch['image'].to(device)
                labs = batch['label'].to(device)

                loss, pred = model(imgs, labs)
                eval_loss += loss.item() * imgs.shape[0] / eval_size
                all_pred.append(pred)
                all_labels.append(labs)

            all_pred = torch.cat(all_pred)
            all_labels = torch.cat(all_labels)
            map_value = metric(all_pred, all_labels)
            print(f"Evaluation loss: {eval_loss}")
            print(f"Mean average precision: {map_value}")

    print("\nTraining took {:.2f} seconds\n".format(time.time() - start))
    torch.save(model.state_dict(), os.path.join(save_dir, save_file_name))


def test():
    model.load_state_dict(torch.load(os.path.join(save_dir, save_file_name)))
    model.eval()
    test_loss = 0
    test_pred = []
    test_labels = []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['image'].to(device)
            labs = batch['label'].to(device)

            loss, pred = model(imgs, labs)
            test_loss += loss.item() * imgs.shape[0] / test_size
            test_pred.append(pred)
            test_labels.append(labs)

    test_pred = torch.cat(test_pred)
    test_labels = torch.cat(test_labels)
    map_value = metric(test_pred, test_labels)
    print(f"Test loss: {test_loss}")
    print(f"Mean average precision: {map_value}")


def main():
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        print("Invalid mode.")


if __name__ == '__main__':
    main()
