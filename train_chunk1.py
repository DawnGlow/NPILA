# -*- coding:utf-8 -*-
import os.path
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from loss import *
from tools import *


def train_model(num1, X_profiling, Y_profiling, X_test, Y_test, model, save_file_name, learning_rate, epochs, batch_size):
    check_file_exists(os.path.dirname(save_file_name))
    file_path = save_file_name + 'model_weights_' + str(num1) + '.h5'
    
    # Save model every epoch
    save_model = ModelCheckpoint(save_best_only=True, filepath=file_path, overwrite=True, mode='auto', save_freq='epoch')

    with tf.device('/GPU:0'):
        # Sanity check
        Reshaped_X_profiling = X_profiling.reshape(X_profiling.shape[0], X_profiling.shape[1], 1)
        Reshaped_X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        callbacks = [save_model]
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
        history = model.fit(x=Reshaped_X_profiling, y=to_categorical(Y_profiling, num_classes=2),
                            validation_data=(Reshaped_X_test, to_categorical(Y_test, num_classes=2)),
                            batch_size=batch_size, verbose=1, epochs=epochs, callbacks=callbacks)

    return history


def train_model_st(source_data_profiling, source_labels_profiling, target_data_profiling, source_data_test, source_labels_test,
                   model, save_file_name, folder, epochs, batch_size, learning_rate, lambda_mmd, num_classes=2):
    """Train a CNN with MMD loss for domain adaptation."""
    check_file_exists(os.path.dirname(save_file_name))
    file_path = save_file_name + 'model_weights_mmd1.h5'

    with tf.device('/GPU:0'):
        # Reshape inputs to match CNN input shape
        Reshaped_source_data_profiling = source_data_profiling.reshape((source_data_profiling.shape[0], source_data_profiling.shape[1], 1))
        Reshaped_source_data_test = source_data_test.reshape((source_data_test.shape[0], source_data_test.shape[1], 1))
        Reshaped_target_data_profiling = target_data_profiling.reshape((target_data_profiling.shape[0], target_data_profiling.shape[1], 1))

        # Define loss, metrics, optimizer
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy()
        val_accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy()
        optimizer = Adam(learning_rate=learning_rate)
        best_val_loss = float("inf")

        history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }

        for epoch in range(epochs):
            for i in range(0, len(Reshaped_source_data_profiling), batch_size):
                # Get a batch from source domain
                source_batch = Reshaped_source_data_profiling[i:i + batch_size]
                source_labels_batch = source_labels_profiling[i:i + batch_size]
                
                # Random batch from target domain
                target_batch = Reshaped_target_data_profiling[
                    np.random.choice(len(Reshaped_target_data_profiling), batch_size)]

                with tf.GradientTape() as tape:
                    # Forward pass
                    source_logits = model(source_batch, training=True)
                    target_logits = model(target_batch, training=True)

                    # Compute classification loss and MMD loss
                    classification_loss = loss_fn(source_labels_batch, source_logits)
                    mmd_loss = compute_mmd_loss(source_logits, target_logits)
                    total_loss = classification_loss + mmd_loss * lambda_mmd

                # Backward pass and optimization
                gradients = tape.gradient(total_loss, model.trainable_weights)
                optimizer.apply_gradients(zip(gradients, model.trainable_weights))
                accuracy_fn.update_state(source_labels_batch, source_logits)

            # Training metrics
            train_accuracy = accuracy_fn.result().numpy()
            print(f"Epoch {epoch + 1}/{epochs}, classification loss: {classification_loss.numpy()}, MMD loss: {mmd_loss.numpy()}, train accuracy: {train_accuracy}", end=", ")

            # Validation step
            val_logits = model(Reshaped_source_data_test, training=False)
            val_loss = loss_fn(source_labels_test, val_logits)
            val_accuracy_fn.update_state(source_labels_test, val_logits)
            val_accuracy = val_accuracy_fn.result().numpy()
            val_accuracy_fn.reset_states()

            print(f"Validation loss: {val_loss.numpy()}, Validation accuracy: {val_accuracy}")

            history['loss'].append(classification_loss.numpy())
            history['val_loss'].append(val_loss.numpy())
            history['accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_weights(file_path)

        # Save all training/validation history to files
        loss_file = folder + '/loss.txt'
        val_loss_file = folder + '/val_loss.txt'
        accuracy_file = folder + '/accuracy.txt'
        val_accuracy_file = folder + '/val_accuracy.txt'

        np.savetxt(loss_file, history['loss'])
        np.savetxt(val_loss_file, history['val_loss'])
        np.savetxt(accuracy_file, history['accuracy'])
        np.savetxt(val_accuracy_file, history['val_accuracy'])

    return history
