#!/usr/bin/env python3
"""
    A python script that optimizes a MobileNetV3-Small model using GPyOpt.
"""
import GPyOpt
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

tf.get_logger().setLevel('ERROR')
tf.random.set_seed(42)
np.random.seed(42)
AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 10
IMG_SIZE = 96

RAW_TRAIN = tfds.load("stl10", split="train[:80%]", as_supervised=True)
RAW_VAL = tfds.load("stl10", split="train[80%:]", as_supervised=True)
RAW_TEST = tfds.load("stl10", split="test", as_supervised=True)


def preprocess(image, label):
    """
        Preprocesses an image and label for training.

        Args:
            image (tf.Tensor): a raw image tensor from the dataset.
            label (int): an integer label for the image class.

        Returns:
            A tuple containing:
                - preprocessed_image: an image tensor scaled to [-1, 1].
                - one_hot_label: a one-hot encoded label vector.
    """
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = preprocess_input(image * 255.0)
    label = tf.one_hot(label, NUM_CLASSES)

    return image, label


def augment(image, label):
    """
        Applies data augmentation to an image.

        Args:
            image (tf.Tensor): a raw image tensor from the dataset.
            label (int): the One-hot encoded label.

        Returns:
            A tuple containing:
                - image: the image with random augmentations applied.
                - label: the One-hot encoded label.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    return image, label


def make_pipelines(batch_size):
    """
        Creates optimized data pipelines for training, validation, and testing.

        Args:
            batch_size (int): the number of samples per batch.

        Returns:
            A tuple containing:
                - dataset_train: the training dataset with augmentation,
                    shuffled and batched
                - dataset_val: the validation dataset, batched and prefetched
                - dataset_test: the test dataset, batched and prefetched
    """
    dataset_train = RAW_TRAIN.map(preprocess,
                                  num_parallel_calls=AUTOTUNE)
    dataset_train = dataset_train.map(augment,
                                      num_parallel_calls=AUTOTUNE)
    dataset_train = dataset_train.shuffle(
        5000,
        seed=42,
        reshuffle_each_iteration=True).batch(batch_size).prefetch(AUTOTUNE)

    dataset_val = RAW_VAL.map(
        preprocess,
        num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    dataset_test = RAW_TEST.map(
        preprocess,
        num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)

    return dataset_train, dataset_val, dataset_test


def mobilenetv3_small(dropout_top, dense_units, l2_weight, unfreeze_ratio):
    """
        Builds a MobileNetV3-Small model with custom classification head.

        Args:
            dropout_top (float): the dropout rate for the classification head.
            dense_units (int): the number of units in the dense layer before
                output.
            l2_weight (float): a L2 regularization weight for the dense layer.
            unfreeze_ratio (float): the proportion of base model layers to
                unfreeze.

        Returns:
            A tuple containing:
                - model: the complete Keras Model ready for training.
                - base: the MobileNetV3-Small base model (initially frozen).
                - n_unfreeze: the number of layers to unfreeze during
                    fine-tuning.
                - total_layers: the total number of layers in the base model.
    """
    base = MobileNetV3Small(include_top=False,
                            weights="imagenet",
                            input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    if dropout_top > 0:
        x = layers.Dropout(dropout_top)(x)
    x = layers.Dense(int(dense_units),
                     activation="relu",
                     kernel_regularizer=regularizers.l2(l2_weight))(x)
    outputs = layers.Dense(NUM_CLASSES,
                           activation="softmax")(x)
    model = Model(inputs, outputs)

    total_layers = len(base.layers)
    n_unfreeze = max(1, int(float(unfreeze_ratio) * total_layers))

    return model, base, n_unfreeze, total_layers


RESULTS_DIR = "mnv3_stl10_bayes_results_1"
os.makedirs(RESULTS_DIR, exist_ok=True)

HEAD_EPOCHS = 5
FT_EPOCHS = 10


def append_log(iteration, params, history_head=None, history_ft=None,
               checkpt_file=None):
    """
        Writes training results to a log file for each Bayesian optimization
        iteration.

        Args:
            iteration (int): the current Bayesian optimization iteration
                number.
            params (str): a string representation of hyperparameters used.
            history_head (History, optional): a Keras History object from
                head training.
            history_ft (History, optional): a Keras History object from
                fine-tuning.
            checkpt_file (str, optional): the path to saved model checkpoint.
    """
    log_path = os.path.join(RESULTS_DIR, "bayes_opt.txt")
    with open(log_path, "a") as f:
        f.write(f"\n=== Iteration {iteration} ===\n")
        f.write(f"Parameters: {params}\n\n")

        best_val_acc = 0.0

        if history_head:
            f.write("[Head Training]\n")
            for i in range(len(history_head.history['loss'])):
                acc = history_head.history['accuracy'][i]
                loss = history_head.history['loss'][i]
                val_acc = history_head.history.get(
                    'val_accuracy',
                    [None]*len(history_head.history['loss']))[i]
                val_loss = history_head.history.get(
                    'val_loss',
                    [None]*len(history_head.history['loss']))[i]

                val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "NA"
                val_loss_str = f"{val_loss:.4f}"\
                    if val_loss is not None else "NA"

                f.write(f"Epoch {i+1}: accuracy={acc:.4f}, loss={loss:.4f}, "
                        f"val_accuracy={val_acc_str}, val_loss={val_loss_str}"
                        f"\n")

                if val_acc is not None:
                    best_val_acc = max(best_val_acc, val_acc)

        if history_ft:
            f.write("\n[Fine-Tuning]\n")
            for i in range(len(history_ft.history['loss'])):
                acc = history_ft.history['accuracy'][i]
                loss = history_ft.history['loss'][i]
                val_acc = history_ft.history.get(
                    'val_accuracy',
                    [None]*len(history_ft.history['loss']))[i]
                val_loss = history_ft.history.get(
                    'val_loss',
                    [None]*len(history_ft.history['loss']))[i]

                val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "NA"
                val_loss_str = f"{val_loss:.4f}"\
                    if val_loss is not None else "NA"

                f.write(f"Epoch {i+1}: accuracy={acc:.4f}, loss={loss:.4f}, "
                        f"val_accuracy={val_acc_str}, val_loss={val_loss_str}"
                        f"\n")

                if val_acc is not None:
                    best_val_acc = max(best_val_acc, val_acc)

        f.write(f"\nBest val_accuracy : {best_val_acc:.4f}\n")

        if checkpt_file:
            f.write(f"\n[Checkpoint] Model saved at: {checkpt_file}\n")


def save_checkpoint(model, iteration, lr_head, lr_ft, dropout_rate, l2_weight,
                    batch_size, dense_units, unfreeze_ratio, results_dir):
    """
        Saves a checkpoint of the best iteration during each training session.
        Specifies the values of the hyperparameters tuned in the filename of
        the checkpoint.

        Args:
            model (Model): the trained Keras model to save.
            iteration (int): the current Bayesian optimization iteration
                number.
            lr_head (float): the learning rate used for head training.
            lr_ft (float): Learning rate used for fine-tuning.
            dropout_rate (float): the dropout rate applied in the model.
            l2_weight (float): the L2 regularization weight.
            batch_size (int): the batch size used during training.
            dense_units (int): the number of units in the dense layer.
            unfreeze_ratio (float): the proportion of layers unfrozen.
            results_dir (str): the directory path to save checkpoints.

        Returns:
            The path to the saved checkpoint file.
    """
    params_str = (f"lr_head{lr_head:.6f}_lr_ft{lr_ft:.6f}_"
                  f"dropout{dropout_rate:.3f}_"
                  f"l2_reg{l2_weight:.6f}_"
                  f"batch_size{int(batch_size)}_"
                  f"dense_units{int(dense_units)}_"
                  f"unfreeze{unfreeze_ratio:.3f}")
    run_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(run_dir, exist_ok=True)
    checkpt_file = os.path.join(run_dir,
                                f"iter{iteration}_model_{params_str}.keras")
    model.save(checkpt_file)

    return checkpt_file


def train_once(lr_head, lr_ft, dropout_rate, l2_weight, batch_size,
               dense_units, unfreeze_ratio, iteration):
    """
        Performs one complete training cycle with given hyperparameters.
        Executes a two-phase training process: head-only training phase and
        fine-tuning phase.

        Args:
            lr_head (float): the learning rate for head-only training phase.
            lr_ft (float): the learning rate for fine-tuning phase.
            dropout_rate (float): the dropout rate for the classification head.
            l2_weight (float): the L2 regularization weight for dense layers.
            batch_size (int): the number of samples per training batch.
            dense_units (int): the number of units in the dense layer.
            unfreeze_ratio (float): the proportion of base model layers to
                unfreeze.
            iteration (int): the current iteration number for logging.

        Returns:
            The best validation accuracy achieved across both training phases.
    """
    ds_train, ds_val, ds_test = make_pipelines(int(batch_size))
    model, base, n_unfreeze, total_layers = mobilenetv3_small(dropout_rate,
                                                              int(dense_units),
                                                              l2_weight,
                                                              unfreeze_ratio)

    print(f"Iteration {iteration} : training head-only")
    model.compile(optimizer=Adam(learning_rate=lr_head),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    callbacks_head = [EarlyStopping(monitor="val_accuracy",
                                    patience=3,
                                    restore_best_weights=True,
                                    verbose=1),
                      ReduceLROnPlateau(monitor="val_loss",
                                        factor=0.5,
                                        patience=2,
                                        min_lr=1e-5,
                                        verbose=1),]
    hist_head = model.fit(ds_train,
                          validation_data=ds_val,
                          epochs=HEAD_EPOCHS,
                          callbacks=callbacks_head,
                          verbose=1)
    best_head_val_acc = float(np.max(hist_head.history["val_accuracy"]))
    head_epochs_run = len(hist_head.history["loss"])
    print(f"[Head] Best val_accuracy: {best_head_val_acc:.4f}")

    for layer in base.layers[-n_unfreeze:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=lr_ft),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    callbacks_ft = [EarlyStopping(monitor="val_accuracy",
                                  patience=4,
                                  restore_best_weights=True,
                                  verbose=1),
                    ReduceLROnPlateau(monitor="val_loss",
                                      factor=0.5,
                                      patience=4,
                                      min_lr=1e-7,
                                      verbose=1),]
    print(f"Iteration {iteration} : Fine-Tuning "
          f"(unfreezing last {n_unfreeze}/{total_layers} layers)")
    hist_ft = model.fit(ds_train,
                        validation_data=ds_val,
                        epochs=FT_EPOCHS,
                        callbacks=callbacks_ft,
                        verbose=1)
    best_ft_val_acc = float(np.max(hist_ft.history["val_accuracy"]))
    ft_epochs_run = len(hist_ft.history["loss"])
    print(f"[Fine-Tuning] Best val_accuracy: {best_ft_val_acc:.4f}")

    final_val_acc = max(best_head_val_acc, best_ft_val_acc)

    checkpt_file = save_checkpoint(model,
                                   iteration,
                                   lr_head,
                                   lr_ft,
                                   dropout_rate,
                                   l2_weight,
                                   batch_size,
                                   dense_units,
                                   unfreeze_ratio,
                                   RESULTS_DIR)

    params_str = (f"lr_head={lr_head:.6f}, lr_ft={lr_ft:.6f}, "
                  f"dropout={dropout_rate:.3f}, "
                  f"l2_reg={l2_weight:.6f}, "
                  f"batch_size={int(batch_size)}, "
                  f"dense_units={int(dense_units)}, "
                  f"unfreeze={unfreeze_ratio:.3f}")
    append_log(iteration,
               params_str,
               history_head=hist_head,
               history_ft=hist_ft,
               checkpt_file=checkpt_file)

    return final_val_acc


iteration_counter = 0


def objective(hp_array):
    """
        The objective function for Bayesian optimization.
        Evaluates a set of hyperparameters by training the model.

        Args:
            hp_array (np.ndarray): a 2D array containing hyperparameter
                values in order: [lr_head, lr_ft, dropout_rate, l2_weight,
                batch_size, dense_units, unfreeze_ratio].

        Returns:
            A 2D array containing (1 - validation_accuracy) for minimization.
    """
    global iteration_counter
    iteration_counter += 1
    print(f"\n=== BO Iteration {iteration_counter} ===")

    (lr_head, lr_ft, dropout_rate, l2_weight,
     batch_size, dense_units, unfreeze_ratio) = hp_array[0]

    print(f"Trying params -> lr_head={lr_head:.6f}, lr_ft={lr_ft:.6f}, "
          f"dropout_rate={dropout_rate:.3f}, "
          f"l2_reg={l2_weight:.6f}, "
          f"batch_size={int(batch_size)}, "
          f"dense_units={int(dense_units)}, "
          f"unfreeze_rate={unfreeze_ratio:.3f}")

    val_acc = train_once(
        lr_head=float(lr_head),
        lr_ft=float(lr_ft),
        dropout_rate=float(dropout_rate),
        l2_weight=float(l2_weight),
        batch_size=int(batch_size),
        dense_units=int(dense_units),
        unfreeze_ratio=float(unfreeze_ratio),
        iteration=iteration_counter)
    print(f"Iteration {iteration_counter} -> val_acc={val_acc:.4f}\n")

    return np.array([[1.0 - val_acc]])


bounds = [
    {'name': 'lr_head', 'type': 'continuous', 'domain': (5e-4, 1e-3)},
    {'name': 'lr_ft',   'type': 'continuous', 'domain': (1e-5, 5e-4)},
    {'name': 'dropout', 'type': 'continuous', 'domain': (0.0, 0.3)},
    {'name': 'l2_reg',  'type': 'continuous', 'domain': (1e-2, 1e-1)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (64, 128)},
    {'name': 'dense_units', 'type': 'discrete', 'domain': (16, 32)},
    {'name': 'unfreeze_ratio', 'type': 'continuous', 'domain': (0.01, 0.2)},]


if __name__ == "__main__":
    init_points = len(bounds)
    optimizer = GPyOpt.methods.BayesianOptimization(f=objective,
                                                    domain=bounds,
                                                    acquisition_type='EI',
                                                    acquisition_jitter=0.05,
                                                    exact_feval=True,
                                                    initial_design_numdata=5)
    optimizer.run_optimization(max_iter=25)
    print(f"Best hyperparameters (x_opt): {optimizer.x_opt}")
    print(f"Best objective (1 - val_accuracy): {optimizer.fx_opt}")
    print(f"Best val_accuracy: {1.0 - optimizer.fx_opt:.4f}")

    with open(os.path.join(RESULTS_DIR, "bayes_opt.txt"), "a") as f:
        f.write("\n=== Bayesian Optimization Report ===\n")
        f.write(f"Best hyperparameters (x_opt): {optimizer.x_opt}\n")
        f.write(f"Best objective (1 - val_accuracy): {optimizer.fx_opt}\n")
        f.write(f"Best val_accuracy: {1.0 - optimizer.fx_opt}\n")
        f.write(str(optimizer.get_evaluations()) + "\n")

    optimizer.plot_convergence()
    plt.savefig(os.path.join(RESULTS_DIR, "convergence.png"))
    plt.close('all')
