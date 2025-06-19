#!/usr/bin/env python3
"""
    A script that trains a MobileNetV2 model to classify the CIFAR 10 dataset.
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt


def preprocess_data(X, Y, batch_size=64, is_training=True):
    """
        Pre-processes the data for the model.

        Args:
            X (np.array): a numpy.ndarray of shape (m, 32, 32, 3) containing
                the CIFAR 10 data, where m is the number of data points.
            Y (np.array): a numpy.ndarray of shape (m,) containing the CIFAR 10
                labels for X.
            batch_size (int): Size of the batches.
            is_training (bool): Whether to apply data augmentation.

        Returns:
            A batched and preprocessed dataset.
    """
    def resize_and_preprocess(img, label):
        """
            Resizes the image to match MobileNetV2 expected input
            and applies preprocessing.

            Args:
                img (tf.Tensor): An image tensor.
                label (tf.Tensor): A label tensor.

            Returns:
                The processed image and one-hot encoded label.
        """
        img = tf.image.resize(img, (224, 224))
        if is_training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_contrast(img, 0.9, 1.1)
            img = tf.image.random_crop(
                tf.image.resize_with_crop_or_pad(img, 230, 230), [224, 224, 3])
        img = preprocess_input(img)
        label = tf.one_hot(label[0], 10)

        return img, label

    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if is_training:
        ds = ds.shuffle(5000)
    ds = ds.map(resize_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds


def build_model(trainable=False):
    """
        Builds and compiles a MobileNetV2-based image classification model.

        Args:
            trainable (bool): Train the base model if True. False as default.

        Returns:
            The full model and the base MobileNetV2 model.
    """
    base_model = MobileNetV2(include_top=False,
                             input_shape=(224, 224, 3),
                             weights='imagenet')
    base_model.trainable = trainable

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model, base_model


if __name__ == '__main__':
    from tensorflow.keras.datasets import cifar10

    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    train_ds = preprocess_data(X_train, Y_train, is_training=True)
    val_ds = preprocess_data(X_test, Y_test, is_training=False)

    model, base_model = build_model(trainable=False)

    def lr_schedule(epoch):
        """
            Computes a linear decays learning rate.

            Args:
                epoch (int): the current epoch.

            Returns:
                A new learning rate.
        """
        new_lr = 0.001 - (epoch * 0.0001)
        return max(new_lr, 0.0001)

    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=3,
                                   restore_best_weights=True,
                                   verbose=1)

    history = model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        callbacks=[early_stopping, lr_scheduler]
    )

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,
                                  patience=2,
                                  min_lr=1e-6,
                                  verbose=1)

    fine_tune_history = model.fit(
        train_ds,
        epochs=11,
        validation_data=val_ds,
        callbacks=[early_stopping, reduce_lr]
    )

    model.save('cifar10.h5')
