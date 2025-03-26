import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# Load UTK Face dataset
def load_data(data_dir):
    images = []
    for filename in os.listdir(data_dir):
        img = tf.keras.preprocessing.image.load_img(os.path.join(data_dir, filename), target_size=(64, 64))
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        images.append(img)
    return np.array(images)

# Define the generator model
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((8, 8, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
    return model

# Define the discriminator model
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

# Compile the models
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# Training the GAN
def train_gan(epochs, batch_size, data_dir):
    dataset = load_data(data_dir)
    for epoch in range(epochs):
        noise = tf.random.normal([batch_size, 100])
        generated_images = generator(noise)
        real_images = dataset[np.random.randint(0, dataset.shape[0], size=batch_size)]

        # Train the discriminator
        with tf.GradientTape() as disc_tape:
            real_output = discriminator(real_images)
            fake_output = discriminator(generated_images)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator.optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        # Train the generator
        noise = tf.random.normal([batch_size, 100])
        with tf.GradientTape() as gen_tape:
            generated_images = generator(noise)
            fake_output = discriminator(generated_images)
            gen_loss = generator_loss(fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator.optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

# Define loss functions
def discriminator_loss(real_output, fake_output):
    return tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output) + \
           tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)

def generator_loss(fake_output):
    return tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output)

# Example usage
data_directory = 'path_to_utkface_dataset'
train_gan(epochs=10000, batch_size=64, data_dir=data_directory)

