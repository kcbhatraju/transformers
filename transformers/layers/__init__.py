import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim, max_seq_len, dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x):
        # x: (batch_size, seq_len)

        # embed: (batch_size, seq_len, embed_dim)
        # multiply by sqrt(embed_dim) so positional
        # encoding doesn't dominate the embedding
        embed = self.embedding(x)
        embed *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))

        # mask: (batch_size, seq_len)
        mask = self.embedding.compute_mask(x)
        
        # indices: (d/2,)
        # indices = [1/10000^(2i/d) for i in range(d/2)]
        indices = tf.range(self.embed_dim // 2, dtype=tf.float32)
        indices = tf.pow(1./10000., 2 * indices / self.embed_dim)

        # indices: (max_seq_len, d/2)
        # indices[i] = [1/10000^(2j/d) for j in range(d/2)]
        indices = tf.repeat(tf.expand_dims(indices, axis=0), self.max_seq_len, axis=0)

        # positions: (max_seq_len, 1)
        # positions = [0, 1, 2, ..., max_seq_len-1]^T
        positions = tf.range(self.max_seq_len, dtype=tf.float32)
        positions = tf.expand_dims(positions, axis=-1)

        # angles: (max_seq_len, d/2)
        # broadcasts positions along last axis to shape (max_seq_len, d/2)
        # such that positions[i] = [i, i, i, ..., i]
        # angles[i] = [i/10000^(2j/d) for j in range(d/2)]
        angles = positions * indices

        # pos_encoding: (1, max_seq_len, d)
        # sin and cos arrays apply elementwise: (max_seq_len, d/2)
        # this is concatenated along last axis to yield encoding
        # extra axis is added to allow batch broadcasting
        # pos_encoding[i] =
        # [sin(i/10000^(2j/d)) for j in range(d/2)] +
        # [cos(i/10000^(2j/d)) for j in range(d/2)]
        # in the original paper, the sin and cos are alternated,
        # but this formulation is equivalent (order doesn't matter)
        pos_encoding = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)

        # final_encoding: (batch_size, seq_len, d)
        # add (batch_size, seq_len, d) + (1, seq_len, d)
        # broadcasts pos_encoding along batch dimension 
        curr_seq_len = tf.shape(embed)[1]
        final_encoding = embed + pos_encoding[:, :curr_seq_len]
        final_encoding = self.dropout(final_encoding)

        return final_encoding, mask

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, embed_dim, key_dim=None, value_dim=None, dropout_rate=0.1):
        super().__init__()

        if key_dim is None:
            key_dim = embed_dim // num_heads
        
        if value_dim is None:
            value_dim = key_dim
        
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.Wq = self.add_weight(shape=(num_heads, embed_dim, key_dim))
        self.Wk = self.add_weight(shape=(num_heads, embed_dim, key_dim))
        self.Wv = self.add_weight(shape=(num_heads, embed_dim, value_dim))
        self.Wo = self.add_weight(shape=(num_heads * value_dim, embed_dim))

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def attention_block(self, query, key, value, padding_mask=None, use_causal_mask=False):
        scores = tf.matmul(query, key, transpose_b=True)
        scores /= tf.math.sqrt(tf.cast(self.key_dim, tf.float32))

        if padding_mask:
            padding_mask = padding_mask[:, None, None, :]
            scores += (1. - padding_mask) * -1e9
        
        if use_causal_mask:
            causal_mask = tf.linalg.band_part(tf.ones_like(scores), -1, 0)
            scores += (1. - causal_mask) * -1e9
        
        alphas = tf.nn.softmax(scores, axis=-1)
        attention = tf.matmul(alphas, value)
        attention_shape = tf.shape(attention)

        attention = tf.reshape(attention, (attention_shape[0], attention_shape[2], self.num_heads * self.value_dim))
        return attention
    
    def call(self, x, padding_mask=None, use_causal_mask=False):
        Pq, Pk, Pv = x
        
        query = tf.matmul(Pq[:, None], self.Wq[None, :])
        key = tf.matmul(Pk[:, None], self.Wk[None, :])
        value = tf.matmul(Pv[:, None], self.Wv[None, :])

        dot_attention = self.attention_block(query, key, value, padding_mask, use_causal_mask)
        concat_attention = tf.matmul(dot_attention, self.Wo[None, :])
        final_attention = self.dropout(concat_attention)

        return final_attention
    
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, embed_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()

        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
            tf.keras.layers.Dropout(dropout_rate)
        ])
    
    def call(self, x):
        return self.mlp(x)
