import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from config import *
from utils import *
from model import *
data,info=tfds.load('mnist',with_info=True)
train_data=data['train']
train_dataset = train_data.map(normalize).shuffle(buffer_size , reshuffle_each_iteration=True).batch(batch_size)
disc = Discriminator()
gen = Generator(img_size)
gan = GAN(discriminator=disc, generator=gen, latent_dim=latent_dim)
gan.compile(
    disc_optimizer=disc_optimizer,
    gen_optimizer=gen_optimizer,
    loss_fn=cross_entropy,
    generator_loss = generator_loss,
    discriminator_loss = discriminator_loss
)
training_callback = TrainingCallback(10, saving_rate)
gan.fit(
    train_dataset, 
    epochs=epochs,
    callbacks=[training_callback]
)
training_callback = TrainingCallback(10, saving_rate)
gan.fit(
    train_dataset,
    epochs=epochs,
    callbacks=[training_callback]
)
disc.save_weights('./models/discriminator')
gen.save_weights('./models/generator')



