from .utils import _DecoderBlock


class PostLayerNormDecoderBlock(_DecoderBlock):
    def call(self, x, encoder_output, padding_mask=None, encoder_padding_mask=None):
        x = self.layernorm1(x + self.self_attention((x, x, x), padding_mask))
        x = self.layernorm2(x + self.cross_attention((x, encoder_output, encoder_output), encoder_padding_mask))
        x = self.layernorm3(x + self.feedforward(x))
        return x

class PreLayerNormDecoderBlock(_DecoderBlock):
    def call(self, x, encoder_output, padding_mask=None, encoder_padding_mask=None):
        norm1 = self.layernorm1(x)
        x = x + self.self_attention((norm1, norm1, norm1), padding_mask)
        norm2 = self.layernorm2(x)
        x = x + self.cross_attention((norm2, encoder_output, encoder_output), encoder_padding_mask)
        norm3 = self.layernorm3(x)
        x = x + self.feedforward(norm3)
        return x

class B2TDecoderBlock(_DecoderBlock):
    def call(self, x, encoder_output, padding_mask=None, encoder_padding_mask=None):
        x2 = self.layernorm1(x + self.self_attention((x, x, x), padding_mask))
        x3 = self.layernorm2(x2 + self.cross_attention((x2, encoder_output, encoder_output), encoder_padding_mask))
        x = self.layernorm3(x + x3 + self.feedforward(x3))
        return x

class ResiDualDecoderBlock(_DecoderBlock):
    def call(self, x, encoder_output, padding_mask=None, encoder_padding_mask=None):
        post_x, pre_x = x

        attn = self.self_attention((post_x, post_x, post_x), padding_mask)
        post_x = self.layernorm1(post_x + attn)
        pre_x = pre_x + attn

        attn = self.cross_attention((post_x, encoder_output, encoder_output), encoder_padding_mask)
        post_x = self.layernorm2(post_x + attn)
        pre_x = pre_x + attn

        ff = self.feedforward(post_x)
        post_x = self.layernorm3(post_x + ff)
        pre_x = pre_x + ff

        return post_x, pre_x
