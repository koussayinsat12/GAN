from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import time
warnings.filterwarnings("ignore")
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.flatten=Flatten()
        self.dense_1=Dense(512,activation='leaky_relu')
        self.batch_1=BatchNormalization(momentum=0.8)
        self.dense_2=Dense(256,activation='leaky_relu')
        self.batch_2=BatchNormalization(momentum=0.8)
        self.dense_3=Dense(128,activation='leaky_relu')
        self.batch_3=BatchNormalization(momentum=0.8)
        self.dense_4=Dense(1,activation='sigmoid')
    def call(self,inputs):
        x=self.flatten(inputs)
        x=self.dense_1(x)
        x=self.batch_1(x)
        x=self.dense_2(x)
        x=self.batch_2(x)
        x=self.dense_3(x)
        x=self.batch_3(x)
        return self.dense_4(x)
class Generator(tf.keras.Model):
    def __init__(self,img_size):
        super(Generator,self).__init__()
        self.dense_1=Dense(128,activation='leaky_relu')
        self.batch_1=BatchNormalization()
        self.dense_2=Dense(256,activation='leaky_relu')
        self.batch_2=BatchNormalization()
        self.dense_3=Dense(512,activation='leaky_relu')
        self.batch_3=BatchNormalization()
        self.dense_4=Dense(np.prod(img_size))
        self.reshape=Reshape(img_size)
    def call(self,inputs):
        x=self.dense_1(inputs)
        x=self.batch_1(x)
        x=self.dense_2(x)
        x=self.batch_2(x)
        x=self.dense_3(x)
        x=self.batch_3(x)
        x=self.dense_4(x)
        return self.reshape(x)

class GAN(tf.keras.Model):
    def __init__(self,discriminator,generator,latent_dim):
        super(GAN,self).__init__()
        self.discriminator=discriminator
        self.generator=generator
        self.latent_dim=latent_dim
    def compile(self,disc_optimizer,gen_optimizer,generator_loss,discriminator_loss,loss_fn):
        super(GAN,self).compile()
        self.disc_optimizer=disc_optimizer
        self.gen_optimizer=gen_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.loss_fn = loss_fn

    @tf.function
    def train_step(self,images):
        batch_size=tf.shape(images)[0]
        noise=tf.random.normal([batch_size,self.latent_dim])

        with tf.GradientTape(persistent=True) as tape:
            generated_images=self.generator(noise)
            real_output=self.discriminator(images)
            fake_output=self.discriminator(generated_images)
            #loss    
            gen_loss = self.generator_loss(self.loss_fn, fake_output)
            disc_loss = self.discriminator_loss(self.loss_fn, real_output, fake_output)
        #calculate gradient
        grad_disc=tape.gradient(disc_loss,self.discriminator.trainable_variables)
        grad_gen=tape.gradient(gen_loss,self.generator.trainable_variables)

        #optimisation
        self.disc_optimizer.apply_gradients(zip(grad_disc,self.discriminator.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(grad_gen,self.generator.trainable_variables))

        return {"Gen Loss" : gen_loss,"Disc Loss": disc_loss}

class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, latent_dim, saving_rate):
        super(TrainingCallback, self).__init__()
        self.latent_dim = latent_dim
        self.saving_rate = saving_rate

    # Save Image sample from Generator
    def save_imgs(self, epoch):
        # Number of images = 16
        seed = tf.random.normal([16, self.latent_dim])
        gen_imgs = self.model.generator(seed)

        fig = plt.figure(figsize=(4, 4))

        for i in range(gen_imgs.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(gen_imgs[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        save_images_dir = "./images"
        if not os.path.exists(save_images_dir):
            os.makedirs(save_images_dir)
        fig.savefig("images/mnist_%d.png" % epoch)

    # Called after each epoch
    def on_epoch_end(self, epoch, logs=None):
        # Save image after 50 epochs
        if epoch % 50 == 0:
            self.save_imgs(epoch)

        if epoch > 0 and epoch % self.saving_rate == 0:
            save_dir = "./models/model_epoch_" + str(epoch)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.model.discriminator.save_weights(save_dir + '/discriminator_%d' % epoch)
            self.model.generator.save_weights(save_dir + '/generator_%d' % epoch)

        self.best_weights = self.model.get_weights()