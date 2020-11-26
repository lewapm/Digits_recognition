import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import tqdm
import numpy as np

from torch.utils.data import DataLoader
from typing import Optional
from sklearn.metrics import accuracy_score, f1_score
from dataset import Dataset
from data_reader import prepare_input

DEVICE = "cpu"

def prepare_model(layer_dims, linear_dims, activation_fn):
    modules_list = []
    for layer_idx in range(len(layer_dims[:-1])):
        spatial_layer = nn.Conv2d(
            layer_dims[layer_idx][0]["channels"], layer_dims[layer_idx + 1][0]["channels"],
            kernel_size=layer_dims[layer_idx + 1][0]["kernel_size"],
            stride=layer_dims[layer_idx + 1][0]["stride"],
            padding=layer_dims[layer_idx + 1][0]["padding"]
        )
        if layer_dims[layer_idx + 1][1] is not None:
            max_pool_kernel = layer_dims[layer_idx + 1][1]["kernel_size"]
            max_pool_stride = layer_dims[layer_idx + 1][1]["stride"]
            modules_list += [activation_fn(), spatial_layer, nn.MaxPool2d(max_pool_kernel, stride=max_pool_stride)]
        else:
            modules_list += [activation_fn(), spatial_layer]
    modules_list += [nn.Flatten()]
    for idx, x in enumerate(linear_dims[1:-1]):
        modules_list += [nn.Linear(linear_dims[idx], x), activation_fn()]
    modules_list += [nn.Linear(linear_dims[-2], linear_dims[-1])]
    model = nn.Sequential(*modules_list).to(DEVICE)
    return model

#
# class (nn.Module):
#     def __init__(self, input_dim, hidden_dim, batch_size, tagset_size):
#         super(LSTMRecognizer, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.batch_size = batch_size
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#
#         self.linear1 = nn.Linear(hidden_dim, 200)
#         self.linear2 = nn.Linear(200, tagset_size)
#         self.hidden = self.reset_hidden()
#
#     def reset_hidden(self):
#         return (torch.zeros(1, self.batch_size, self.hidden_dim),
#                        torch.zeros(1, self.batch_size, self.hidden_dim))
#
#     def forward(self, sent, sent_len):
#         self.hidden = self.reset_hidden()
#         X = torch.nn.utils.rnn.pack_padded_sequence(sent, sent_len, batch_first=True, enforce_sorted=False)
#         _, self.hidden = self.lstm(X, self.hidden)
#         x = self.linear1(self.hidden[0])
#         x = F.relu(x)
#         x = self.linear2(x)
#         tag_scores = F.softmax(x, dim=2)
#         return tag_scores


class ImageClassifier:
    def __init__(self):
        self.model=None

    def prepare_model(self, training_params):
        self.model = prepare_model(training_params['layer_dims'], training_params['linear_dims'], training_params['activation_fn'])

    def train_epoch(self, train_loader, batch_size):
        loss_fn = nn.CrossEntropyLoss()
        y_true = []
        y_pred = []
        for batch_idx, sample in enumerate(tqdm.tqdm(train_loader)):
            self.optimizer.zero_grad()
            X = sample['image']
            y = sample['label']
            preds = self.model.forward(X)
            preds = preds.view(batch_size, -1)
            loss = loss_fn(preds, y)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            y_pred += torch.argmax(preds, dim=1).tolist()
            y_true += y.tolist()
        f1 = f1_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)
        return f1, acc

    def eval_model(self, valid_loader, batch_size):
        y_true = []
        y_pred = []
        for batch_idx, sample in enumerate(valid_loader):
            X = sample['image']
            y = sample['label']
            preds = self.model.forward(X)
            preds = preds.view(batch_size, -1)
            y_pred += torch.argmax(preds, dim=1).tolist()
            y_true += y.tolist()
        f1 = f1_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)
        return f1, acc

    def train(self, train_dataset: Dataset, validation_dataset: Dataset, training_params: Optional[dict]) -> None:
        print(training_params)
        train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True, drop_last=True)
        valid_loader = DataLoader(validation_dataset, batch_size=training_params['batch_size'], shuffle=True,
                                  drop_last=True)

        if training_params['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=training_params["lr"])
        elif training_params['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=training_params["lr"])

        train_f1, train_acc, valid_f1, valid_acc = 0, 0, 0, 0
        best_acc = 0
        for epoch in range(training_params['epochs']):
            train_f1, train_acc = self.train_epoch(train_loader, training_params['batch_size'])
            print(f"After epoch {epoch} for train acc: {train_acc:.3f}  f1: {train_f1:.3f}")
            valid_f1, valid_acc = self.eval_model(valid_loader, training_params['batch_size'])
            print(f"After epoch {epoch} for valid acc: {valid_acc:.3f}  f1: {valid_f1:.3f}")


    # def predict(self, utterance: str) -> dict:
    #     if self.embedder is None or self.model is None:
    #         raise ('Load or train model, before predict')
    #     sent = normalize(utterance)
    #     sent = self.embedder.create_sentence_embedding(sent, predict=True)
    #     sent = torch.unsqueeze(sent, 0)
    #     with torch.no_grad():
    #         x = self.model.forward(sent, torch.tensor([sent.shape[1]])).view(-1)
    #         predicted = torch.argmax(x).tolist()
    #         return {'intent': self.labels_dict[predicted], 'confidence': x[predicted].tolist()}


hyperparams = {
    "batch_size": 64,
    "layer_dims": [[{"channels": 1, "kernel_size": 3, "stride": 1, "padding": 1}, None],
                   [{"channels": 16, "kernel_size": 3, "stride": 1, "padding": 1}, {"kernel_size": 2, "stride": 2}],
                   [{"channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}, {"kernel_size": 2, "stride": 2}],
                   [{"channels": 64, "kernel_size": 3, "stride": 1, "padding": 1}, {"kernel_size": 2, "stride": 2}]],
    "linear_dims": [3136, 500, 36],
    "activation_fn": nn.ReLU,
    'optimizer': "Adam",
    "lr": 1e-3,
    'epochs': 10,
    "seed": 1
}

np.random.seed(hyperparams["seed"])
torch.manual_seed(hyperparams["seed"])

train_img, train_lab, valid_img, valid_lab = prepare_input()
train_dataset = Dataset(train_img, train_lab)
valid_dataset = Dataset(valid_img, valid_lab)
model = ImageClassifier()
model.prepare_model(hyperparams)
model.train(train_dataset, valid_dataset, hyperparams)
