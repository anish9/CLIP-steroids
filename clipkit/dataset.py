from typing import Literal
import numpy as np
import pandas as pd
import tensorflow as tf
from pydantic import FilePath
from transformers import PreTrainedTokenizer

size_h, size_w = 288, 288


def random_tile_mask(image, mask_size_h=32, mask_size_w=32):
    h, w, c = (size_h, size_w, 3)
    mask_size = tf.minimum(mask_size_h, tf.minimum(h, w))
    top = tf.random.uniform([], 0, h - mask_size_h, dtype=tf.int32)
    left = tf.random.uniform([], 0, w - mask_size_w, dtype=tf.int32)
    mask = tf.ones([mask_size_h, mask_size_w, c], dtype=image.dtype)
    pad_top = top
    pad_bottom = h - (top + mask_size_h)
    pad_left = left
    pad_right = w - (left + mask_size_w)
    full_mask = tf.pad(
        mask, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0
    )
    return image * (1 - full_mask)


def random_cutout(x, min_h=30, max_h=80, min_w=30, max_w=80):
    x = random_tile_mask(
        x,
        mask_size_h=tf.random.uniform([], min_h, max_h, dtype=tf.int32),
        mask_size_w=tf.random.uniform([], min_w, max_w, dtype=tf.int32),
    )
    return x


def augument_im(x):
    x = tf.cond(
        tf.random.uniform([], 0, 1) > 0.7,
        lambda: tf.image.random_flip_left_right(x),
        lambda: x,
    )
    x = tf.cond(
        tf.random.uniform([], 0, 1) > 0.01,
        lambda: tf.image.random_crop(x, (size_h, size_w, 3)),
        lambda: tf.image.resize_with_pad(x, size_h, size_w),
    )
    x = tf.cond(tf.random.uniform([], 0, 1) > 0.01, lambda: random_cutout(x), lambda: x)
    return x


def data_process(tokenizer: PreTrainedTokenizer, text_max_len=12):
    def data_preprocess(image_path: tf.Tensor, text: tf.Tensor, mode: str = "train"):
        def _tokenize_py(text):
            if hasattr(text, "numpy"):
                text = text.numpy()
            if isinstance(text, bytes):
                txt = text.decode("utf-8")
            elif isinstance(text, np.ndarray):
                txt = text.item().decode("utf-8")
            else:
                txt = str(text)

            enc = tokenizer(
                txt,
                padding="max_length",
                truncation=True,
                max_length=text_max_len,
                return_tensors="np",
            )
            return enc["input_ids"][0], enc["attention_mask"][0]

        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)

        if mode == "train":
            img = tf.image.resize_with_pad(img, 300, 300)
            img = augument_im(img)
        elif mode == "test":
            img = tf.image.resize_with_pad(img, size_h, size_w)

        input_ids, attention_mask = tf.py_function(
            func=_tokenize_py, inp=[text], Tout=[tf.int32, tf.int32]
        )
        input_ids.set_shape([text_max_len])
        attention_mask.set_shape([text_max_len])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": img,
        }

    return data_preprocess


def load_data(
    data_id: FilePath,
    tokenizer: PreTrainedTokenizer,
    text_max_len: int,
    mode: Literal["train", "test"],
    batch_size: int,
):
    data = pd.read_csv(data_id)
    data_images = data.image.tolist()[:]
    data_texts = data.text.tolist()[:]
    data_set = tf.data.Dataset.from_tensor_slices((data_images, data_texts))
    base_tokenizer_func = data_process(tokenizer=tokenizer, text_max_len=text_max_len)
    proc_data = data_set.map(
        lambda x, y: base_tokenizer_func(image_path=x, text=y, mode=mode),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    proc_data = proc_data.batch(batch_size, drop_remainder=True)
    return proc_data
