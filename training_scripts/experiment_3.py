import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
from pathlib import Path

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define the DCGAN generator model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(11 * 14 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((11, 14, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Additional Conv2DTranspose layers to achieve target size
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 1), padding='valid', use_bias=False, activation='tanh'))  # Corrected output shape

    return model


# Define the DCGAN discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[355, 228, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Define the latent space mutation function
def mutate_latent_space(latent_vector, mutation_rate):
    mutated_vector = latent_vector + mutation_rate * np.random.randn(*latent_vector.shape)
    return mutated_vector

# Define the training loop
def train_dcgan(generator, discriminator, dataset, epochs, batch_size, latent_dim, mutation_rate):
    # Define loss functions and optimizers
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # Track genotype variations
    genotype_variations = []

    @tf.function
    def train_step(images, latent_vector):
        # Generate noise for fake images
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake images
            generated_images = generator(latent_vector, training=True)

            # Discriminator loss calculation
            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)
            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
            disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + \
                        cross_entropy(tf.zeros_like(fake_output), fake_output)

        # Update generator and discriminator
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss

    for epoch in range(epochs):
        for images in dataset:
            # Normalize images to [-1, 1]
            # Generate random latent vectors
            latent_vectors = tf.random.normal([batch_size, latent_dim])

            # Perform latent space mutation
            mutated_latent_vectors = mutate_latent_space(latent_vectors, mutation_rate)

            # Train the DCGAN using original and mutated latent vectors
            gen_loss, disc_loss = train_step(images, latent_vectors)
            mutated_gen_loss, mutated_disc_loss = train_step(images, mutated_latent_vectors)

            # Track genotype variations
            genotype_variations.append(mutated_latent_vectors)

        # Display progress every epoch
        print(f'Epoch {epoch + 1}/{epochs}, Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}, Mut. Generator Loss: {gen_loss:.4f}, Mut. Discriminator Loss: {disc_loss:.4f}')

        # Generate a sample image every 5 epochs
        if (epoch + 1) % 5 == 0:
            sample_latent_vector = tf.random.normal([1, latent_dim])
            generated_image = generator(sample_latent_vector, training=False)
            generated_image = (generated_image[0] + 1.0) / 2.0  # De-normalize to [0, 1]
            plt.imshow(generated_image)
            plt.axis('off')
            fig = plt.gcf()
            fig.savefig(f"generator_model/{Path(__file__).stem}/{epoch + 1}.png")

    return genotype_variations

# Prepare the dataset (folder containing JPEG images)
if not os.path.exists(f"generator_model/{Path(__file__).stem}"):
    os.mkdir(f"generator_model/{Path(__file__).stem}")

epochs = 50
batch_size = 32
latent_dim = 100
mutation_rate = 0.12

def preprocess_image(image, label):
    # Resize and normalize the images
    image = tf.image.resize(image, (355, 228))
    image = (image - 127.5) / 127.5  # Normalize to [-1, 1]
    return image

# Prepare the dataset (folder containing JPEG images)
data_path = '/home/breimers/Pictures/img_align_celeba/train/celeb'
image_files = tf.data.Dataset.list_files(data_path + '/*.jpg', shuffle=True)

# Define a function to read and preprocess the image files
def read_and_preprocess_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = preprocess_image(image, label=None)  # We don't need labels
    print("Image shape:", image.shape)  # Add this line to print the image shape
    return image

# Map the image file paths to the read_and_preprocess_image function
train_dataset = image_files.map(read_and_preprocess_image)

# Shuffle and batch the dataset
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

print("Built Dataset")
# Set hyperparameters

# Create the generator and discriminator models
generator = make_generator_model()
discriminator = make_discriminator_model()

# Train the DCGAN with latent space mutation and track genotype variations
genotype_variations = train_dcgan(generator, discriminator, train_dataset, epochs, batch_size, latent_dim, mutation_rate)
generator.save(f"generator_model/{Path(__file__).stem}")
np.save(f"generator_model/{Path(__file__).stem}/genotype_variations.npy", genotype_variations)