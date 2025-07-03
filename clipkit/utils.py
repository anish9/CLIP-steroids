import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import __dict__ as keras_models
from transformers import AutoTokenizer, TFAutoModel


def gpu_memory_fix():
    """Avoids wasting GPU memory in the import of tensorflow"""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def get_image_model(model_name: str, active_layers_image_model: int) -> Model:
    """Loads a pre-trained model and freezes initial layers for transfer learning.
    Args:
        model_name: The name of the Keras model to load. (efficientnet, resnet, densenet etc...)
        active_layers_image_model: Number of final layers to keep trainable.
    Returns:
        Keras Model
    """
    try:
        ImageModel = keras_models[model_name](include_top=False, weights="imagenet")
        for layer in ImageModel.layers[:-active_layers_image_model]:
            layer.trainable = False

        return ImageModel
    except Exception as e:
        raise RuntimeError(f"Failed to load image model: '{model_name}'") from e


def get_text_model(model_id: str, trainable: bool) -> TFAutoModel:
    """Loads a Huggingface pre-trained text model and sets training signature.
    Args:
        model_id: huggingface text models id
        trainable : bool
    Returns:
        TFAutoModel
    """
    try:
        model = TFAutoModel.from_pretrained(model_id, from_pt=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model.trainable = trainable
        return model, tokenizer
    except Exception as e:
        raise RuntimeError("Failed to load text model") from e


class CheckpointSaver(tf.keras.callbacks.Callback):
    """customized model ckpt callback"""

    def __init__(self, ckpt_dir=None, max_to_keep=3, verbose=False):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.max_to_keep = max_to_keep
        self.verbose = verbose
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def set_model(self, model):
        """One time set-up"""
        super().set_model(model)
        self.ckpt = tf.train.Checkpoint(
            net=model, optimizer=model.optimizer, epoch=tf.Variable(0, dtype=tf.int64)
        )
        self.manager = tf.train.CheckpointManager(
            checkpoint=self.ckpt, directory=self.ckpt_dir, max_to_keep=self.max_to_keep
        )

        latest = self.manager.latest_checkpoint
        if latest:
            _ = model(
                {
                    "pixel_values": tf.ones((1, 224, 224, 3)),
                    "input_ids": tf.ones((1, 12), tf.int32),
                    "attention_mask": tf.ones((1, 12), tf.int32),
                }
            )
            self.ckpt.restore(latest).expect_partial()
            print(f"restored from {latest}")

    def on_epoch_end(self, epoch, logs=None):
        self.ckpt.epoch.assign_add(epoch + 1)
        self.manager.save()
        if self.verbose:
            print(f" : Saved checkpoint for epoch {epoch + 1} at {self.ckpt_dir}")
