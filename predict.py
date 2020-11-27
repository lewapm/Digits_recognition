import torch
import numpy as np

from dataset import Dataset
from data_reader import prepare_input
from model import ImageClassifier
from hyperparams import hyperparams

seed = hyperparams["seed"]
np.random.seed(seed)
torch.manual_seed(seed)


def create_and_train_model(path='./train.pkl'):
    train_img, train_lab, valid_img, valid_lab = prepare_input(path)
    train_dataset = Dataset(train_img, train_lab)
    valid_dataset = Dataset(valid_img, valid_lab)
    model = ImageClassifier()
    model.prepare_model(hyperparams)
    model.train(train_dataset, valid_dataset, hyperparams)
    return model


def get_model(path):
    try:
        model = torch.load(path + '/weights.ckpt')
    except FileNotFoundError:
        model = create_and_train_model()
    return model


def predict(input_data=None):
    model = get_model(hyperparams['model_save_dir'])
    preds = model.forward(input_data)
    preds = torch.argmax(preds, dim=1).tolist()
    return preds

predict()