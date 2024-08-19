import tensorflow as tf
from blocks.decoders import (B2TDecoderBlock, PostLayerNormDecoderBlock,
                             PreLayerNormDecoderBlock, ResiDualDecoderBlock)
from blocks.encoders import (B2TEncoderBlock, PostLayerNormEncoderBlock,
                             PreLayerNormEncoderBlock, ResiDualEncoderBlock)
from layers import PositionalEncoding


class _Section(tf.keras.Model):
    def __init__(self, section_type, num_blocks, num_heads, vocab_size, embed_dim, hidden_dim, max_seq_len, dropout_rate=0.1, block_type="ResiDual"):
        super().__init__()
        self.positional = PositionalEncoding(vocab_size, embed_dim, max_seq_len, dropout_rate)

        self.blocks = [globals()[f"{block_type}{section_type}Block"](num_heads, embed_dim, hidden_dim, dropout_rate) for _ in range(num_blocks)]

        if block_type == "PreLayerNorm" or block_type == "ResiDual":
            self.final_layernorm = tf.keras.layers.LayerNormalization()
        
        if section_type == "Decoder":
            self.final_dense = tf.keras.layers.Dense(vocab_size)
        
        self.section_type = section_type
        self.block_type = block_type
        
    def build(self, *args):
        return
    
    def call(self, x, *args, **kwargs):
        x, padding_mask = self.positional(x)

        if self.block_type == "ResiDual":
            x = (x, x)

        for block in self.blocks:
            x = block(x, *args, padding_mask, **kwargs)
        
        if self.block_type == "PreLayerNorm":
            x = self.final_layernorm(x)
        elif self.block_type == "ResiDual":
            x = x[0] + self.final_layernorm(x[1])
        
        if self.section_type == "Encoder":
            return x, padding_mask
        elif self.section_type == "Decoder":
            x = self.final_dense(x)
            return x