import tensorflow as tf


class Warmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embed_dim, num_warmup=4000):
        super().__init__()
        self.embed_dim = tf.cast(embed_dim, tf.float32)
        self.num_warmup = tf.cast(num_warmup, tf.float32)
    
    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        return tf.math.rsqrt(self.embed_dim) * tf.math.minimum(tf.math.rsqrt(step), step * (self.num_warmup ** -1.5))
