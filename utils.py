import matplotlib.pyplot as plt
import numpy as np
from data_reader import load_data, augmentation_pic


def plot_training_loss(loss_tab, name='loss'):
    plt.figure()
    plt.plot(loss_tab)
    plt.savefig('./'+name+'.jpg')
    plt.close()


def plot_two_images(wr, cor, label_wr, label_cor):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(wr.reshape(56, 56), cmap='gray', interpolation='none')
    plt.axis("off")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(cor.reshape(56, 56), cmap='gray', interpolation='none')
    plt.axis("off")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"./plots/{label_cor}_{label_wr}")
    plt.close()


def plot_classes_samples():
    images, labels = load_data()
    res = [None] * 36
    for i in range(len(images)):
        if res[labels[i][0]] is None:
            res[labels[i][0]] = images[i]

    plt.figure(figsize=(12, 6))
    for i in range(36):
        plt.subplot(6, 6, i + 1)
        xdd = augmentation_pic(res[i])
        plt.imshow(xdd[2].reshape(56, 56), cmap='gray', interpolation='none')
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig('./plots/all_classes')
    plt.close()


def get_classes_sizes():
    images, labels = load_data()
    unique, counts = np.unique(labels, return_counts=True)
    labels_counter = dict(zip(unique, counts))
    return labels_counter


def get_correct_incorrect_predictions(images, labels, predict):
    wrong = [None] * 36
    correct = [None] * 36
    for i in range(len(images)):
        if labels[i] != predict[i] and wrong[labels[i]] is None:
            wrong[labels[i]] = (images[i], predict[i])
        if labels[i] == predict[i] and correct[labels[i]] is None:
            correct[labels[i]] = images[i]
    return wrong, correct
