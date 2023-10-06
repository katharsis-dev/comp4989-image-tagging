import torch.nn as nn
import torchmetrics as tm
import torch


class MultiLabelClassifier(nn.Module):
    def __init__(self, model, p_drop, n_model_out, n_classes):
        super().__init__()
        self.cnn = model
        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(n_model_out, n_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.metric = tm.AveragePrecision('multilabel', num_labels=n_classes)

    def forward(self, image, label):
        output = self.cnn(image).logits
        output = self.dropout(output)
        logits = self.fc(output)
        loss = self.loss_fn(logits, label.float())
        pred = torch.sigmoid(logits)
        MAP = self.metric(pred, label)

        return loss, MAP
