import torch.nn as nn

hyperparams = {
    "seed": 1,
    "batch_size": 128,
    "optimizer": "Adam",
    "lr": 1e-3,
    'epochs': 5,
    "layer_dims": [[{"channels": 1, "kernel_size": 3, "stride": 1, "padding": 1}, None, 'n'],
                   [{"channels": 16, "kernel_size": 3, "stride": 1, "padding": 1}, {"kernel_size": 2, "stride": 2}, 'r'],
                   [{"channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}, {"kernel_size": 2, "stride": 2}, 'r'],
                   [{"channels": 64, "kernel_size": 3, "stride": 1, "padding": 1}, {"kernel_size": 2, "stride": 2}, 'n']],
    "linear_dims": [3136, 500, 36],
    "activation_fn": nn.ReLU,
    "l1_reg": 0.0,
    "model_save_dir": './trained_model',
}