import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

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
    img_15 = ndimage.rotate(pat, 45, reshape=False).reshape(-1)
    img_m15 = ndimage.rotate(pat, -15, reshape=False).reshape(-1)
    img_30 = ndimage.rotate(pat, 30, reshape=False).reshape(-1)
    img_m30 = ndimage.rotate(pat, -37, reshape=False).reshape(-1)
    return img_15, img_30, img_m15, img_m30


def create_augmented_list(images):
    res = []
    for img in images:
        a, b, c, d = augmentation_pic(img)
        res.append(a)
        res.append(b)
        res.append(c)
        res.append(d)
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
        if len(img_lists[i]) < 100: continue
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


def prepare_input():
    images, labels = load_data()
    train_img, train_lab, valid_img, valid_lab = create_test_valid_ds(images, labels, 0.15)
    return train_img, train_lab, valid_img, valid_lab

#
# a, b, c ,d = prepare_input()
# print(len(a), len(b))

# images, labels = load_data()
# res = [None] * 36
# for i in range(len(images)):
#     if res[labels[i][0]] is None:
#         res[labels[i][0]] = images[i]
#
# print(len(res))
# plt.figure(figsize=(12, 6))
# for i in range(36):
#     # print(i)
#
#     plt.subplot(6, 6, i + 1)
#     # plt.tight_layout()
#     plt.imshow(res[i].reshape(56, 56), cmap='gray', interpolation='none')
#     plt.axis("off")
#     plt.xticks([])
#     plt.yticks([])
#
# plt.subplots_adjust(hspace=0, wspace=0)
# plt.show()
#
#
# #
# images, labels = load_data()
# unique, counts = np.unique(labels, return_counts=True)
# labels_counter = dict(zip(unique, counts))
# print(labels_counter)
# x = np.array(sorted(counts))
# print(np.sum(x[1:18])*0.85*4)
# # tmp = np.reshape(res[6], (56, 56))
# # plt.figure()
# # plt.imshow(tmp, cmap='gray')
# # plt.show()
# #
# #
# print(sorted(counts))




