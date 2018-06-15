import numpy as np
import cv2
import random
from PIL import Image
import os
from astropy.visualization import MinMaxInterval
interval = MinMaxInterval()

random.seed(777)

real_train_path = '/Users/lastland/datasets/FictureFind/sample_train/real_img/'
fake_train_path = '/Users/lastland/datasets/FictureFind/sample_train/fake_img/'

real_test_path = '/Users/lastland/datasets/FictureFind/sample_test/real_img/'
fake_test_path = '/Users/lastland/datasets/FictureFind/sample_test/fake_img_dc/'

real_train_list = os.listdir(real_train_path)
fake_train_list = os.listdir(fake_train_path)

real_test_list = os.listdir(real_test_path)
fake_test_list = os.listdir(fake_test_path)


def make_label(batch_size, keywords) : # make the labels for real or fake images

    if keywords == 'real' or 1:
        real_label = np.array([[1, 0]] * batch_size)  # (batch_size, 1) [Real, Fake]
        return real_label

    elif keywords == 'fake' or 0 :
        fake_label = np.array([[0, 1]] * batch_size)  # (batch_size, 1), [Real, Fake]
        return fake_label

'''
def make_real_label(batch_size) : # make the labels for real image --> 1

    real_label = [[1, 0]] * batch_size # (batch_size, 1) [Real, Fake]

    return real_label



def make_fake_label(batch_size) : # make the labels for fake image --> 0

    fake_label = [[0, 1]] * batch_size # (batch_size, 1), [Real, Fake]

    return fake_label'''



def read_real_img(batch_count, batch_size, path) : # read the real images

    real_img = []
    list = os.listdir(path)

    for content in list[batch_count : (batch_count + batch_size)] :

        img = Image.open('%s%s' % (path, content))
        img = np.array(img.crop((0,20,178,198)))
        img = cv2.resize(img, (63, 63), interpolation=cv2.INTER_AREA)
        real_img.append(img)

    return real_img



def read_fake_img(batch_count, batch_size, path) : # read the fake images

    fake_img = []
    list = os.listdir(path)

    for content in list[batch_count : (batch_count + batch_size)] :

        img = np.array(Image.open('%s%s' % (path, content)))
        img = cv2.resize(img, (63,63), interpolation=cv2.INTER_AREA)
        fake_img.append(img)

    return fake_img



def make_train_batch(batch_index, batch_size) : # make the batch for train inputs

    batch_count = batch_index * batch_size

    real_img = read_real_img(batch_count, batch_size, path=real_train_path)
    fake_img = read_fake_img(batch_count, batch_size, path=fake_train_path)
    batch_img = np.concatenate((real_img, fake_img))

    real_label = make_label(batch_size, keywords='real')
    fake_label = make_label(batch_size, keywords='fake')
    batch_label = np.concatenate((real_label, fake_label))

    shuffle_img = []
    shuffle_label = []
    random_indexes = random.sample(range(0, batch_size * 2), batch_size * 2)

    for index in random_indexes :

        shuffle_img.append(batch_img[index])
        shuffle_label.append(batch_label[index])

    return shuffle_img, shuffle_label


def load_one_image(path, content):
    fake_img = []
    img = np.array(Image.open('%s%s' % (path, content)))
    img = cv2.resize(img, (63,63), interpolation=cv2.INTER_AREA)

    fake_img.append(img)
    return fake_img


def make_test_batch() : # make the batch for test inputs

    total_real_test = len(real_test_list)
    total_fake_test = len(fake_test_list)

    real_img = read_real_img(batch_count=0, batch_size=total_real_test, path=real_test_path)
    fake_img = read_fake_img(batch_count=0, batch_size=total_fake_test, path=fake_test_path)
    batch_img = np.concatenate((real_img, fake_img))

    real_label = make_label(batch_size=total_real_test, keywords='real')
    fake_label = make_label(batch_size=total_fake_test, keywords='fake')
    batch_label = np.concatenate((real_label, fake_label))

    shuffle_img = []
    shuffle_label = []
    random_indexes = random.sample(range(0, total_real_test * 2), total_real_test * 2)

    for index in random_indexes:

        shuffle_img.append(batch_img[index])
        shuffle_label.append(batch_label[index])

    return shuffle_img, shuffle_label
