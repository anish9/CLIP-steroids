import tensorflow as tf
import yaml

tf.config.optimizer.set_jit(False)  # jit error fix for text
from pprint import pprint

from tensorflow.keras.callbacks import TensorBoard

from clipkit import gpu_memory_fix
from clipkit.cliplayers import ClipMe
from clipkit.dataset import load_data
from clipkit.utils import CheckpointSaver, get_image_model, get_text_model

gpu_memory_fix()

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

pprint(cfg)
data_cfg = cfg["Dataset"]
model_cfg = cfg["Model"]
train_cfg = cfg["Training"]


image_model = get_image_model(
    model_name=model_cfg["image_encoder"],
    active_layers_image_model=model_cfg["tune_image_layers_count"],
)


text_model, tokenizer = get_text_model(
    model_id=model_cfg["text_encoder"], trainable=model_cfg["tune_text_layers"]
)

train_data = load_data(
    data_id=data_cfg["train_csv_path"],
    tokenizer=tokenizer,
    text_max_len=model_cfg["max_length"],
    mode="train",
    batch_size=train_cfg["batch_size"],
)

val_data = load_data(
    data_id=data_cfg["val_csv_path"],
    tokenizer=tokenizer,
    text_max_len=model_cfg["max_length"],
    mode="test",
    batch_size=train_cfg["batch_size"],
)

CLIPME = ClipMe(
    image_model_id=image_model, text_model_id=text_model, proj_dim=model_cfg["proj_dim"]
)

optimizer = tf.keras.optimizers.Adam(float(train_cfg["learning_rate"]))
CLIPME.compile(optimizer=optimizer, jit_compile=False)

tensorboard_callback = TensorBoard(log_dir=train_cfg["logs_dir"])
ckpt_callback = CheckpointSaver(
    ckpt_dir=train_cfg["ckpt_dir"],
    max_to_keep=train_cfg["ckpt_max_keep"],
    verbose=False,
)

if __name__ == "__main__":
    print("--" * 30)
    print(" Training Started " + chr(0x1F600))
    print("--" * 30)
    CLIPME.fit(
        train_data,
        epochs=train_cfg["epochs"],
        validation_data=val_data,
        batch_size=train_cfg["batch_size"],
        callbacks=[ckpt_callback, tensorboard_callback],
    )
