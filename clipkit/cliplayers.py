import tensorflow as tf


class ImageEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, image_model, proj_units):
        super().__init__()
        self.image_model = image_model
        self.proj_units = proj_units
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.proj = tf.keras.layers.Dense(proj_units)

    def call(self, x):
        return self.proj(self.pool(self.image_model(x)))


class TextEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, text_model, proj_units=256):
        super().__init__()
        self.text_model = text_model
        self.proj_units = proj_units
        self.proj = tf.keras.layers.Dense(proj_units)

    def call(self, inp_ids, att_mask):
        inp_ids = tf.cast(inp_ids, tf.int32)
        att_mask = tf.cast(att_mask, tf.int32)
        # data = {"input_ids":inp_ids,"attention_mask":att_mask}
        outputs = self.text_model(
            input_ids=inp_ids, attention_mask=att_mask
        )  # self.text_model(data)
        cls_output = outputs.last_hidden_state[:, 0, :]
        projected = self.proj(cls_output)
        return projected


class ClipMe(tf.keras.models.Model):
    def __init__(self, image_model_id, text_model_id, proj_dim, temperature=0.07, **kwargs):
        super().__init__()
        self.image_model_id = image_model_id
        self.text_model_id = text_model_id
        self.proj_dim = proj_dim
        self.image_encoder = ImageEncoderLayer(
            image_model=self.image_model_id, proj_units=self.proj_dim
        )
        self.text_encoder = TextEncoderLayer(
            text_model=self.text_model_id, proj_units=self.proj_dim
        )
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
        self.temp = temperature

    def compile(self, *, optimizer, jit_compile=False, **kwargs):
        super().compile(optimizer=optimizer, jit_compile=jit_compile, **kwargs)
        self._optimizer = optimizer

    @property
    def metrics(self):
        return [self.loss_tracker, self.val_loss_tracker]

    def clip_loss(self, image_features, text_features):
        image_features = tf.math.l2_normalize(image_features, axis=-1)
        text_features = tf.math.l2_normalize(text_features, axis=-1)
        logits = tf.matmul(image_features, text_features, transpose_b=True) / self.temp

        labels = tf.range(tf.shape(logits)[0])
        loss_i2t = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        loss_t2i = tf.keras.losses.sparse_categorical_crossentropy(
            labels, tf.transpose(logits), from_logits=True
        )
        return (loss_i2t + loss_t2i) / 2

    def siglip_loss(self, image_features, text_features):
        image_features = tf.math.l2_normalize(image_features, axis=-1)
        text_features = tf.math.l2_normalize(text_features, axis=-1)
        logits = tf.matmul(image_features, text_features, transpose_b=True) / self.temp

        # Create one-hot labels for sigmoid BCE
        batch_size = tf.shape(logits)[0]
        labels = tf.eye(batch_size)

        # Sigmoid + BCE on the full similarity matrix
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        return loss

    def call(self, x, training=True):
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        pixel_values = x["pixel_values"]
        im_features = self.image_encoder(pixel_values)
        txt_features = self.text_encoder(input_ids, attention_mask)
        return im_features, txt_features

    def train_step(self, data):
        with tf.GradientTape() as tape:
            im_feat, txt_feat = self(data)
            loss_value = self.clip_loss(im_feat, txt_feat)  # self.siglip_loss(im_feat,txt_feat)
        image_trainables = self.image_encoder.image_model.trainable_variables
        text_trainables = self.text_encoder.text_model.trainable_variables
        clip_trainables = image_trainables + text_trainables
        grads = tape.gradient(loss_value, clip_trainables)
        self._optimizer.apply_gradients(zip(grads, clip_trainables))

        self.loss_tracker.update_state(loss_value)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        with tf.GradientTape() as tape:
            im_feat, txt_feat = self(data)
            loss_value = self.clip_loss(im_feat, txt_feat)  # self.siglip_loss(im_feat,txt_feat)
        self.val_loss_tracker.update_state(loss_value)
        return {"loss": self.val_loss_tracker.result()}

    def get_config(self):
        # start from base config (name, dtype, etc)
        config = super().get_config()
        # then add ours
        config.update(
            {
                "image_model_id": self.image_model_id,
                "text_model_id": self.text_model_id,
                "proj_dim": self.proj_dim,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Keras will pass the same dict you returned from get_config()
        return cls(**config)
