# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow_addons.layers import GroupNormalization
from keras import backend as K
from keras.models import Model
from keras.regularizers import Regularizer
from keras.layers import Input, Conv1D, AveragePooling1D, Flatten, Dense, BatchNormalization, Dropout


class ewc_reg(Regularizer):
    """EWC regularizer to penalize changes to important weights."""
    def __init__(self, fisher, prior_weights, Lambda=0.3):
        self.fisher = fisher
        self.prior_weights = prior_weights
        self.Lambda = Lambda

    def __call__(self, x):
        # Compute EWC penalty
        regularization = self.Lambda * K.sum(self.fisher * K.square(x - self.prior_weights))
        return regularization

    def get_config(self):
        return {'Lambda': float(self.Lambda)}


def cnn_architecture(input_size=450, classes=2):
    """1D CNN model for classification."""
    input_shape = (input_size, 1)
    img_input = Input(shape=input_shape)

    x = Conv1D(8, 1, kernel_initializer='he_uniform', activation='relu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    # x = GroupNormalization(groups=1, axis=-1)(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    x = Conv1D(16, 3, kernel_initializer='he_uniform', activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    # x = GroupNormalization(groups=1, axis=-1)(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)

    x = Flatten(name='flatten')(x)

    x = Dense(200, kernel_initializer='he_uniform', activation='relu', name='fc1')(x)
    x = Dropout(rate=0.3)(x)

    x = Dense(100, kernel_initializer='he_uniform', activation='relu', name='fc2')(x)
    x = Dropout(rate=0.3)(x)

    x = Dense(30, kernel_initializer='he_uniform', activation='tanh', name='fc3')(x)

    x = Dense(classes, activation='softmax', name='predictions')(x)
    
    inputs = img_input
    model = Model(inputs, x, name='block_transfer')

    return model


def cnn_architecture_EWC(I, input_size=450, classes=2, pretrained_model=None):
    """CNN model with Elastic Weight Consolidation (EWC) regularization applied to weights based on a pretrained model."""
    input_shape = (input_size, 1)
    img_input = Input(shape=input_shape)

    x = Conv1D(8, 1, kernel_initializer='he_uniform', activation='relu', padding='same', kernel_regularizer=ewc_reg(I[0],
               pretrained_model.weights[0]), bias_regularizer=ewc_reg(I[1], pretrained_model.weights[1]), name='block1_conv1')(img_input)
    x = BatchNormalization(gamma_regularizer=ewc_reg(I[2], pretrained_model.weights[2]),
                           beta_regularizer=ewc_reg(I[3], pretrained_model.weights[3]))(x)
    # x = GroupNormalization(groups=1, axis=-1)(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    x = Conv1D(16, 3, kernel_initializer='he_uniform', activation='relu', padding='same', kernel_regularizer=ewc_reg(I[6], 
               pretrained_model.weights[6]), bias_regularizer=ewc_reg(I[7], pretrained_model.weights[7]), name='block2_conv1')(x)
    x = BatchNormalization(gamma_regularizer=ewc_reg(I[8], pretrained_model.weights[8]),
                           beta_regularizer=ewc_reg(I[9], pretrained_model.weights[9]))(x)
    # x = GroupNormalization(groups=1, axis=-1)(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)

    x = Flatten(name='flatten')(x)

    x = Dense(200, kernel_initializer='he_uniform', activation='relu', kernel_regularizer=ewc_reg(I[12], pretrained_model.weights[12]),
              bias_regularizer=ewc_reg(I[13], pretrained_model.weights[13]), name='fc1')(x)
    x = Dropout(rate=0.3)(x)

    x = Dense(100, kernel_initializer='he_uniform', activation='relu', kernel_regularizer=ewc_reg(I[14], pretrained_model.weights[14]),
              bias_regularizer=ewc_reg(I[15], pretrained_model.weights[15]), name='fc2')(x)
    x = Dropout(rate=0.3)(x)

    x = Dense(30, kernel_initializer='he_uniform', activation='tanh', kernel_regularizer=ewc_reg(I[16], pretrained_model.weights[16]),
              bias_regularizer=ewc_reg(I[17], pretrained_model.weights[17]), name='fc3')(x)

    x = Dense(classes, activation='softmax', kernel_regularizer=ewc_reg(I[18], pretrained_model.weights[18]),
              bias_regularizer=ewc_reg(I[19], pretrained_model.weights[19]), name='predictions')(x)

    inputs = img_input
    model = Model(inputs, x, name='block_transfer_EWC')

    return model
