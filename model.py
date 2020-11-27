import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from torch.utils.data import DataLoader
from typing import Optional
from sklearn.metrics import accuracy_score, f1_score
from dataset import Dataset
from residual_block import ResidualBlock
from pathlib import Path

DEVICE = "cpu"


def prepare_model(layer_dims, linear_dims, activation_fn):
    modules_list = []
    for layer_idx in range(len(layer_dims[:-1])):
        if layer_dims[layer_idx][2] == 'r':
            r_layer = ResidualBlock(
                channels=layer_dims[layer_idx][0]["channels"],
                kernel_size=layer_dims[layer_idx + 1][0]["kernel_size"],
                stride=layer_dims[layer_idx + 1][0]["stride"],
                padding=layer_dims[layer_idx + 1][0]["padding"],
                activation_fn=activation_fn,
            )
            modules_list += [r_layer]
        layer = nn.Conv2d(
            layer_dims[layer_idx][0]["channels"], layer_dims[layer_idx + 1][0]["channels"],
            kernel_size=layer_dims[layer_idx + 1][0]["kernel_size"],
            stride=layer_dims[layer_idx + 1][0]["stride"],
            padding=layer_dims[layer_idx + 1][0]["padding"]
        )
        modules_list += [layer, activation_fn()]
        if layer_dims[layer_idx + 1][1] is not None:
            max_pool_kernel = layer_dims[layer_idx + 1][1]["kernel_size"]
            max_pool_stride = layer_dims[layer_idx + 1][1]["stride"]
            modules_list += [nn.MaxPool2d(max_pool_kernel, stride=max_pool_stride)]

    modules_list += [nn.Flatten()]
    for idx, x in enumerate(linear_dims[1:-1]):
        modules_list += [nn.Linear(linear_dims[idx], x), activation_fn()]
    modules_list += [nn.Linear(linear_dims[-2], linear_dims[-1])]
    model = nn.Sequential(*modules_list).to(DEVICE)
    return model


class ImageClassifier:
    def __init__(self):
        self.model = None

    def prepare_model(self, training_params):
        self.model = prepare_model(training_params['layer_dims'], training_params['linear_dims'], training_params['activation_fn'])

    def train_epoch(self, train_loader, l1_reg):
        loss_fn = nn.CrossEntropyLoss()
        y_true = []
        y_pred = []
        for batch_idx, sample in enumerate(tqdm.tqdm(train_loader)):
            self.optimizer.zero_grad()
            X = sample['image']
            y = sample['label']
            preds = self.model.forward(X)
            l1_regularization = torch.tensor(0).float()
            for param in self.model.parameters():
                l1_regularization += torch.sum(torch.abs(param))

            loss = loss_fn(preds, y) + l1_reg*l1_regularization
            loss.backward(retain_graph=True)
            self.optimizer.step()
            y_pred += torch.argmax(preds, dim=1).tolist()
            y_true += y.tolist()
        f1 = f1_score(y_true, y_pred, average='weighted')
        acc = accuracy_score(y_true, y_pred)
        return f1, acc

    def eval_model(self, valid_loader):
        y_true = []
        y_pred = []
        for batch_idx, sample in enumerate(valid_loader):
            X = sample['image']
            y = sample['label']
            preds = self.model.forward(X)
            y_pred += torch.argmax(preds, dim=1).tolist()
            y_true += y.tolist()
        f1 = f1_score(y_true, y_pred, average='weighted')
        acc = accuracy_score(y_true, y_pred)
        return f1, acc

    def predict(self, input_data, batch_size):
        predict_loader = DataLoader(input_data, batch_size=batch_size, drop_last=False)
        res_pred = []
        for batch_idx, sample in enumerate(predict_loader):
            X = sample['image']
            preds = self.model.forward(X)
            res_pred += torch.argmax(preds, dim=1).tolist()
        return res_pred

    def train(self, train_dataset, validation_dataset, training_params):
        train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True, drop_last=True)
        valid_loader = DataLoader(validation_dataset, batch_size=training_params['batch_size'], shuffle=True,
                                  drop_last=True)

        if training_params['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=training_params["lr"])
        elif training_params['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=training_params["lr"])

        best_f1 = 0
        for epoch in range(training_params['epochs']):
            train_f1, train_acc = self.train_epoch(train_loader, training_params["l1_reg"])
            print(f"After epoch {epoch} for train acc: {train_acc:.3f}  f1: {train_f1:.3f}")
            valid_f1, valid_acc = self.eval_model(valid_loader)
            print(f"After epoch {epoch} for valid acc: {valid_acc:.3f}  f1: {valid_f1:.3f}")
            if valid_f1 > best_f1:
                self.save_model(training_params['model_save_dir'])
                best_f1 = valid_f1

    def save_model(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        filename = path + "/weights.ckpt"
        torch.save(self.model, filename)

    def load_model(self, path):
        self.model = torch.load(path+'/weights.ckpt')




