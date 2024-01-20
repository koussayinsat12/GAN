import tensorflow as tf
def normalize(x):
    image = tf.cast(x['image'], tf.float32)
    image = (image / 127.5) - 1
    return image
def discriminator_loss(loss_object, real_output, fake_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
def generator_loss(loss_object, discriminator_probability):
    return loss_object(tf.ones_like(discriminator_probability), discriminator_probability)