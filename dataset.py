from torch.utils.data import Dataset
from PIL import Image
import torch


class ImageDataset(Dataset):

    def __init__(self, image_paths, labels, processor):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]).convert("RGB") as image:
            image_t = self.processor(image, return_tensors="pt")
        image_t = image_t.pixel_values.squeeze(0)
        label = torch.tensor(self.labels[idx])

        return {
            'image': image_t,
            'label': label
        }

    def __len__(self):
        return len(self.image_paths)
