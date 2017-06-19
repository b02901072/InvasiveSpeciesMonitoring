#!/usr/bin/env python3

'''
    IO.py: Handle all data input and output
'''
import skimage
import skimage.io
import glob
import os
import numpy as np
import pandas as pd
import csv

def load_train(
        train_img_dirname='data/train_128x128',
        train_label_filename='data/train_labels.csv'
    ):
    train_img = []
    img_filename_list = glob.glob(os.path.join(train_img_dirname, '*.jpg'))
    img_filename_list = sorted(img_filename_list, key=lambda x: int(os.path.basename(x)[:-4]))
    for img_filename in img_filename_list:
        print('Reading', img_filename, end='\r')
        img = skimage.io.imread(img_filename)
        img = np.asarray(img, dtype=np.float)
        train_img.append(img)
        '''
        if train_img is None:
            train_img = img
        else:
            train_img = np.concatenate((train_img, img))
        '''
    train_img = np.asarray(train_img, dtype=np.float) / 255.0
    print('Training Data: ', train_img.shape)

    train_labels = pd.read_csv(train_label_filename, dtype=float)
    train_labels = np.asarray(train_labels)[:, 1]

    return train_img, train_labels


def load_test(
        test_img_dirname='data/test_128x128',
    ):
    test_img = []
    img_filename_list = glob.glob(os.path.join(test_img_dirname, '*.jpg'))
    img_filename_list = sorted(img_filename_list, key=lambda x: int(os.path.basename(x)[:-4]))
    for img_filename in img_filename_list:
        print('Reading', img_filename, end='\r')
        img = skimage.io.imread(img_filename)
        img = np.asarray(img, dtype=np.float)
        test_img.append(img)

    return test_img

def write_result(
        result,
        result_filename,
    ):

    content = 'name,invasive'

    for i in range(result.shape[0]):
        content += "\n" + str(i+1) + "," + str(result[i])

    with open(result_filename, "w") as f:
        f.write(content)
        f.close()


