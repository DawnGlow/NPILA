# -*- coding:utf-8 -*-
import tensorflow as tf


def compute_mmd_loss(source_samples, target_samples, kernel_type='rbf', bandwidth=1.0):
    # Define the Gaussian (RBF) kernel function
    def gaussian_kernel(x, y, bandwidth):
        x_expand = tf.expand_dims(x, 1)  # Shape: (batch_size, 1, num_features)
        y_expand = tf.expand_dims(y, 0)  # Shape: (1, batch_size, num_features)
        return tf.exp(-tf.reduce_sum(tf.square(x_expand - y_expand), axis=-1) / (2 * bandwidth ** 2))

    # Define the linear kernel function
    def linear_kernel(x, y):
        return tf.matmul(x, y, transpose_b=True)

    if kernel_type == 'rbf':
        kernel_function = gaussian_kernel
    elif kernel_type == 'linear':
        kernel_function = linear_kernel
    else:
        raise ValueError("Invalid kernel type")

    # Flatten input samples to 2D tensors (batch_size, features)
    source_samples = tf.reshape(source_samples, [-1, tf.reduce_prod(source_samples.shape[1:])])
    target_samples = tf.reshape(target_samples, [-1, tf.reduce_prod(target_samples.shape[1:])])

    # Compute kernel matrices
    SxS = kernel_function(source_samples, source_samples, bandwidth)  # Source x Source
    SxT = kernel_function(source_samples, target_samples, bandwidth)  # Source x Target
    TxT = kernel_function(target_samples, target_samples, bandwidth)  # Target x Target

    # measures distribution discrepancy
    mmd_loss = tf.reduce_mean(SxS) - 2 * tf.reduce_mean(SxT) + tf.reduce_mean(TxT)
    
    return mmd_loss
