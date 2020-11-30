import pickle
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rotate, AffineTransform, warp


def load_data(path='./train.pkl'):
    with open(path, 'rb') as f:
        images, labels = pickle.load(f)
    return images, labels


def show_pic(pic):
    plt.figure()
    plt.imshow(pic, cmap='gray')
    plt.show()


def augmentation_pic(pic):
    pat = pic.reshape(56, 56)
    tf1 = AffineTransform(shear=0.3)
    tf2 = AffineTransform(shear=-0.3)
    img_1 = rotate(pat, 25).reshape(-1)
    img_2 = rotate(pat, -15).reshape(-1)
    img_3 = warp(pat, tf1, order=1, preserve_range=True)
    img_4 = warp(pat, tf2, order=1, preserve_range=True)
    return [img_1, img_2, img_3, img_4]


def create_augmented_list(images):
    res = []
    for img in images:
        x = augmentation_pic(img)
        res += x
    return res


def create_test_valid_ds(images, labels, valid_size=0.15):
    p = np.random.permutation(len(images))
    images = images[p]
    labels = labels[p]
    img_lists = [[] for _ in range(36)]
    for idx, img in enumerate(images):
        img_lists[labels[idx][0]].append(img)

    train_img, train_lab, valid_img, valid_lab = [], [], [], []
    for i in range(36):
        if len(img_lists[i]) < 100:
            train_img += img_lists[i]
            train_lab += [i] * len(img_lists[i])
            valid_img += img_lists[i]
            valid_lab += [i] * len(img_lists[i])
            continue
        no = len(img_lists[i])
        no_valid = int(no * valid_size)
        train_tmp_img = img_lists[i][:-no_valid]
        if len(img_lists[i]) < 450:
            x = create_augmented_list(train_tmp_img)
            train_tmp_img += x
        train_img += train_tmp_img
        train_lab += [i]*len(train_tmp_img)
        valid_img += img_lists[i][-no_valid:]
        valid_lab += [i] * len(img_lists[i][-no_valid:])
    return train_img, train_lab, valid_img, valid_lab


def prepare_input(path):
    images, labels = load_data(path)
    train_img, train_lab, valid_img, valid_lab = create_test_valid_ds(images, labels, 0.15)
    return train_img, train_lab, valid_img, valid_lab
