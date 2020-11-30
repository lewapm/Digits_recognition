import torch
import numpy as np

from dataset import Dataset
from data_reader import prepare_input
from model import ImageClassifier
from hyperparams import hyperparams
from pathlib import Path

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
        model = ImageClassifier()
        model.load_model(path)
    except FileNotFoundError:
        model = create_and_train_model()
    return model


def predict(input_data=None):
    model = get_model(hyperparams['model_save_dir'])
    if input_data is not None:
        input_dataset = Dataset(input_data, np.zeros(len(input_data)))
        preds = model.predict(input_dataset, hyperparams["batch_size"])
        return np.array(preds).reshape(-1, 1)


Path('./plots').mkdir(parents=True, exist_ok=True)
Path(hyperparams["model_save_dir"]).mkdir(parents=True, exist_ok=True)
_, _, valid_img, valid_lab = prepare_input('./train.pkl')
pred_values = predict(valid_img)
