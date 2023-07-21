import argparse
import tensorflow as tf
import numpy as np
import os
import re
import string
import pickle
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import random

from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from gutenberg.query import get_metadata

# Load text data from files in the given directory
def load_text_data(size):
    texts = []
    titles = []
    for bookid in random.choices(range(1000,30000), k=size):
        print(f"Downloading {bookid}")
        try:
            text = strip_headers(load_etext(bookid)).strip()
            title = str(get_metadata('title', bookid)[0])
            texts.append(text)
            titles.append(title)
        except Exception as e:
            print(e)
    return texts, titles

# Preprocess the text data and generate input and output sequences
def preprocess_text_data(texts, titles, max_sequence_length):
    # Combine title and text to create prompt
    prompts = [title + " " + text for title, text in zip(titles, texts)]

    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(prompts)
    total_words = len(tokenizer.word_index) + 1

    # Create input and output sequences for the LSTM model
    sequences = tokenizer.texts_to_sequences(prompts)
    input_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='pre')
    output_sequences = np.roll(input_sequences, -1, axis=1)
    output_sequences[:, -1] = 0  # Replace last element with 0 to indicate padding
    prompt_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

    return input_sequences, output_sequences, prompt_sequences, total_words, tokenizer

# Build the LSTM-based generator model
def build_generator(total_words, max_sequence_length):
    model = tf.keras.Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_length))
    model.add(LSTM(128))
    model.add(Dense(total_words, activation='softmax'))
    return model

# Build the discriminator model
def build_discriminator(total_words, max_sequence_length):
    model = tf.keras.Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_length))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Train the LSTM-based generator model and the discriminator model
def train_gan(generator, discriminator, combined, input_sequences, output_sequences, prompt_sequences, batch_size, epochs):
    combined.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    for epoch in range(epochs):
        # Train discriminator on real data
        real_labels = np.ones((len(input_sequences), 1))
        d_loss_real = discriminator.train_on_batch(input_sequences, real_labels)

        # Generate fake data using the generator
        fake_data = generator.predict(input_sequences)
        fake_labels = np.zeros((len(input_sequences), 1))
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)

        # Train generator to fool the discriminator
        target_labels = np.ones((len(input_sequences), 1))
        g_loss = combined.train_on_batch(prompt_sequences, output_sequences)

        # Print progress
        print(f"Epoch {epoch + 1}/{epochs}, Discriminator Loss: {0.5 * np.add(d_loss_real, d_loss_fake)[0]}, "
              f"Generator Loss: {g_loss}")

# Generate text using the trained generator model and a specific prompt
def generate_text(generator, tokenizer, seed_prompt_sequence, max_sequence_length, length):
    generated_text = []
    for _ in range(length):  # Generate 1000 words
        generated_sequence = generator.predict(seed_prompt_sequence)
        generated_word_index = np.argmax(generated_sequence[0][-1, :])
        generated_word = tokenizer.index_word[generated_word_index]
        generated_text.append(generated_word)
        seed_prompt_sequence[0] = np.roll(seed_prompt_sequence[0], -1)
        seed_prompt_sequence[0, -1] = generated_word_index
    return " ".join(generated_text)

# Save the generator model and tokenizer
def save_model_and_tokenizer(generator, tokenizer):
    generator.save('generator_model.h5')
    with open('tokenizer.pkl', 'wb') as file:
        pickle.dump(tokenizer, file)

# Load the generator model and tokenizer
def load_model_and_tokenizer():
    generator = load_model('generator_model.h5')
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
    return generator, tokenizer

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Generative Text Model using Adversarial Text Generation")
    parser.add_argument('--mode', choices=['train', 'generate'], required=True, help="Mode: train or generate")
    parser.add_argument('--prompt', help="Text-based prompt for text generation")
    parser.add_argument('--size', help="How many books to pull for training")
    parser.add_argument('--length', help="Length of results")
    return parser.parse_args()

# Main function
if __name__ == "__main__":
    args = parse_args()

    max_sequence_length = 10
    batch_size = 128
    epochs = 20
    length = int(args.length if args.length else 500)
    size = int(args.size if args.size else 1000
)
    if args.mode == 'train':
        # Load and preprocess the text data
        texts, titles = load_text_data(size)
        input_sequences, output_sequences, prompt_sequences, total_words, tokenizer = preprocess_text_data(texts, titles, max_sequence_length)

        # Build and train the discriminator model
        discriminator = build_discriminator(total_words, max_sequence_length)
        discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        discriminator.trainable = False

        # Build the generator model and the combined adversarial model
        generator = build_generator(total_words, max_sequence_length)
        combined_input = Input(shape=(max_sequence_length,), dtype='int32')
        generated_sequences = generator(combined_input)
        validity = discriminator(generated_sequences)
        combined = Model(inputs=combined_input, outputs=[generated_sequences, validity])

        # Train the GAN model
        train_gan(generator, discriminator, combined, input_sequences, output_sequences, prompt_sequences, batch_size, epochs)
        save_model_and_tokenizer(generator, tokenizer)

    elif args.mode == 'generate':
        if args.prompt is None:
            print("Please provide a prompt using the '--prompt' argument for text generation.")
        else:
            generator, tokenizer = load_model_and_tokenizer()
            seed_prompt_sequence = tokenizer.texts_to_sequences([args.prompt])[0]
            seed_prompt_sequence = pad_sequences([seed_prompt_sequence], maxlen=max_sequence_length, padding='pre')
            generated_text = generate_text(generator, tokenizer, seed_prompt_sequence, max_sequence_length, length=length)
            print("Generated Text:")
            print(generated_text)
