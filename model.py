import torch.nn as nn


class MultiLabelClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1000, 81)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.fc(x)
        probs = self.sigmoid(logits)
        return (probs > 0.5).int()
