import tensorflow as tf
from transformers.layers import FeedForward, MultiHeadAttention


class _EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, embed_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, embed_dim, dropout_rate=dropout_rate)
        self.feedforward = FeedForward(embed_dim, hidden_dim, dropout_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()

class _DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, embed_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(num_heads, embed_dim, dropout_rate=dropout_rate)
        self.cross_attention = MultiHeadAttention(num_heads, embed_dim, dropout_rate=dropout_rate)
        self.feedforward = FeedForward(embed_dim, hidden_dim, dropout_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.layernorm3 = tf.keras.layers.LayerNormalization()
