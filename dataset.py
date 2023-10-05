from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
class MoviePosterDataset(Dataset):

    def __init__(self, image_paths, labels, processor):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        # self.processor = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

    def __getitem__(self, idx):
        image = self.image_paths[idx]
        image = Image.open(image).convert("RGB")
        image = self.processor(image, return_tensors="pt")
        image = image.pixel_values.squeeze(0)
        label = torch.tensor(self.labels[idx])

        return {
            'image': image,
            'label': label
        }

    def __len__(self):
        return len(self.image_paths)

