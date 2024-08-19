import tensorflow as tf


def masked_loss(label, logits):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")(label, logits)
    mask = tf.cast((label != 0), dtype=tf.float32)
    loss *= mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def masked_accuracy(label, logits):
    pred = tf.argmax(logits, axis=-1)

    mask = (label != 0)
    correct = (pred == label) & mask

    mask = tf.cast(mask, dtype=tf.float32)
    correct = tf.cast(correct, dtype=tf.float32)

    return tf.reduce_sum(correct) / tf.reduce_sum(mask)
