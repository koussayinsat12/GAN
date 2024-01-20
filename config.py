import tensorflow as tf
latent_dim = 10
epochs = 100
batch_size = 512
saving_rate = 100
buffer_size = 5000
gen_optimizer = tf.keras.optimizers.Adam(0.0001)
disc_optimizer = tf.keras.optimizers.Adam(0.0001)
img_size=(28,28,1)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)