from .utils import _EncoderBlock


class PostLayerNormEncoderBlock(_EncoderBlock):
    def call(self, x, padding_mask=None):
        x = self.layernorm1(x + self.attention((x, x, x), padding_mask))
        x = self.layernorm2(x + self.feedforward(x))
        return x

class PreLayerNormEncoderBlock(_EncoderBlock):
    def call(self, x, padding_mask=None):
        norm1 = self.layernorm1(x)
        x = x + self.attention((norm1, norm1, norm1), padding_mask)
        norm2 = self.layernorm2(x)
        x = x + self.feedforward(norm2)
        return x

class B2TEncoderBlock(_EncoderBlock):
    def call(self, x, padding_mask=None):
        x2 = self.layernorm1(x + self.attention((x, x, x), padding_mask))
        x = self.layernorm2(x + x2 + self.feedforward(x2))
        return x

class ResiDualEncoderBlock(_EncoderBlock):
    def call(self, x, padding_mask=None):
        post_x, pre_x = x
        
        attn = self.attention((post_x, post_x, post_x), padding_mask)
        post_x = self.layernorm1(post_x + attn)
        pre_x = pre_x + attn

        ff = self.feedforward(post_x)
        post_x = self.layernorm2(post_x + ff)
        pre_x = pre_x + ff

        return post_x, pre_x
