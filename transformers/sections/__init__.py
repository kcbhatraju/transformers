from .utils import _Section


class Encoder(_Section):
    def __init__(self, num_blocks, num_heads, vocab_size, embed_dim, hidden_dim, max_seq_len, dropout_rate=0.1, block_type="ResiDual"):
        super().__init__("Encoder", num_blocks, num_heads, vocab_size, embed_dim, hidden_dim, max_seq_len, dropout_rate, block_type)
    
    def call(self, x):
        return super().call(x)

class Decoder(_Section):
    def __init__(self, num_blocks, num_heads, vocab_size, embed_dim, hidden_dim, max_seq_len, dropout_rate=0.1, block_type="ResiDual"):
        super().__init__("Decoder", num_blocks, num_heads, vocab_size, embed_dim, hidden_dim, max_seq_len, dropout_rate, block_type)

    def call(self, x, encoder_output, encoder_padding_mask):
        return super().call(x, encoder_output, encoder_padding_mask=encoder_padding_mask)
