# -*- coding:utf-8 -*-
import os
import os.path
import sys
import random
import numpy as np


def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


def shuffle_data(profiling_x, label_y):
    """Shuffle the input data and corresponding labels in unison."""
    l = list(zip(profiling_x, label_y))
    random.shuffle(l)
    shuffled_x, shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return shuffled_x, shuffled_y


def shuffle_data_4(data1, data2, data3, data4):
    """Shuffle four corresponding data arrays in unison."""
    combined = list(zip(data1, data2, data3, data4))
    random.shuffle(combined)
    shuffled_data = list(zip(*combined))
    shuffled_data = [np.array(d) for d in shuffled_data]
    return shuffled_data[0], shuffled_data[1], shuffled_data[2], shuffled_data[3]


def pearson_correlation(x, y):
    """Compute Pearson correlation"""
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    if np.var(x) == 0 or np.var(y) == 0:
        return 0

    x_std = np.std(x)
    y_std = np.std(y)

    covariance = np.mean((x - x_mean) * (y - y_mean))

    pearson = covariance / (x_std * y_std)

    pearson1 = abs(pearson)

    return pearson1


def load_key(file_path, start=0, length=256):
    """Load n keys starting from line `start` in the file."""
    with open(file_path, 'r') as f:
        line = f.readline().strip()
        key_numbers = list(map(int, line.split()))
        
        selected_keys = key_numbers[start : start + length]
        key = ['Real Keys'] + selected_keys
    
    return key


def save_chunk1_history(history, label, floder, epochs, min_value, max_value, i):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = history.history['val_' + label]
    np.savetxt(floder + 'val_' + label + '(' + str(i) + ')' + ".txt", data['val_' + label])
    np.savetxt(floder + label + '(' + str(i) + ')' + ".txt", data[label])


def save_chunk2_history(history, label, floder, epochs, min_value, max_value, i, j):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = history.history['val_' + label]
    np.savetxt(floder + 'val_' + label + '(' + str(i) + '_' + str(j) + ')' + ".txt", data['val_' + label])
    np.savetxt(floder + label + '(' + str(i) + '_' + str(j) + ')' + ".txt", data[label])
