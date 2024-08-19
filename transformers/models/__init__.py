import tensorflow as tf
from sections import Decoder, Encoder


class Transformer(tf.keras.Model):
    def __init__(self, num_blocks, num_heads, enc_vocab_size, dec_vocab_size, embed_dim, hidden_dim, max_seq_len, dropout_rate=0.1, enc_type="ResiDual", dec_type=None):
        super().__init__()
        if dec_type is None:
            dec_type = enc_type

        self.encoder = Encoder(num_blocks, num_heads, enc_vocab_size, embed_dim, hidden_dim, max_seq_len, dropout_rate, enc_type)
        self.decoder = Decoder(num_blocks, num_heads, dec_vocab_size, embed_dim, hidden_dim, max_seq_len, dropout_rate, dec_type)
    
    def build(self, *args):
        return
    
    def call(self, x):
        encoder_input, decoder_input = x

        encoder_output, encoder_padding_mask = self.encoder(encoder_input)
        decoder_output = self.decoder(decoder_input, encoder_output, encoder_padding_mask)

        return decoder_output
