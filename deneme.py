# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:05:45 2024

@author: sarpa
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer


texts = [
    "The Fulton County Grand Jury said Friday an investigation of Atlanta's recent primary election produced no evidence that any irregularities took place.",
    "The jury further said in term-end presentments that the City Executive Committee , which had over-all charge of the election deserves the praise and thanks of the City of Atlanta for the manner in which the election was conducted.",
    "The September-October term jury had been charged by Fulton Superior Court Judge Durwood Pye to investigate reports of possible irregularities in the hard-fought primary which was won by Mayor-nominate Ivan Allen Jr.",
    "Only a relative handful of such reports was received the jury said considering the widespread interest in the election the number of voters and the size of this city.",
    "The jury said it did find that many of Georgia's registration and election laws are outmoded or inadequate and often ambiguous.",
    "It recommended that Fulton legislators act to have these laws studied and revised to end this problem.",
    "The jury said it believes these two offices should be combined to achieve greater efficiency and reduce the cost of administration.",
    "The City Purchasing Department the jury said is well operated and follows generally accepted practices which inure to the best interest of both the city and its taxpayers.",
    "Regarding the new proposed Public Safety Building the jury said this should be expedited."
]

# Tokenize the texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

# Determine maxlen
maxlen = max(len(seq) for seq in sequences)

# Prepare the training data
X = np.zeros((len(sequences), maxlen, len(word_index) + 1), dtype=np.bool_)
y = np.zeros((len(sequences), len(word_index) + 1), dtype=np.bool_)
for i, seq in enumerate(sequences):
    for t, word_idx in enumerate(seq):
        X[i, t, word_idx] = 1
    y[i, seq[-1]] = 1

# Define the generator model
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100, activation="relu"))
    model.add(Dense(maxlen * (len(word_index) + 1), activation="softmax"))
    model.add(Reshape((maxlen, len(word_index) + 1)))
    return model

# Define the discriminator model
def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(maxlen, len(word_index) + 1)))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model

# Build and compile the GAN
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=["accuracy"])

discriminator.trainable = False

gan_input = tf.keras.Input(shape=(100,))
generated_text = generator(gan_input)
gan_output = discriminator(generated_text)

gan = tf.keras.Model(gan_input, gan_output)
gan.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

# Train the GAN
epochs = 300
batch_size = 32

for epoch in range(epochs):
    # Train the discriminator
    noise = np.random.normal(0, 1, (batch_size, 100))
    generated_texts = generator.predict(noise)
    
    real_texts = X[np.random.randint(0, X.shape[0], batch_size)]
    combined_texts = np.concatenate([generated_texts, real_texts])
    
    labels = np.concatenate([np.zeros((batch_size, 1)), np.ones((batch_size, 1))])
    d_loss = discriminator.train_on_batch(combined_texts, labels)
    
    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    misleading_labels = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, misleading_labels)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}")

# Generate new text
def generate_text():
    noise = np.random.normal(0, 1, (1, 100))
    generated_text = generator.predict(noise)[0]
    generated_text = ' '.join([tokenizer.index_word[np.argmax(char_vec)] for char_vec in generated_text])
    return generated_text

print(generate_text())


