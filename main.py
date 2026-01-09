# import numpy as np
# import tensorflow as tf
# import tensorflow_hub as hub
# import tensorflow_datasets as tfds
# from tensorflow_hub.keras_layer import KerasLayer
#
# train_data, validation_data, test_data = tfds.load(name = "imdb_reviews", split = ('train[:60%]', 'train[60%:]', 'test'), as_supervised=True)
# train_data
# train_example_batch, train_labeled_batch = next(iter(train_data.batch(10)))
# print(train_example_batch)
# print(train_labeled_batch)
# embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
# hub_layer = hub.KerasLayer(embedding, input_shape = [], dtype = tf.string, trainable = True)
# print(hub_layer(train_example_batch[:3]))
#
# model = tf.keras.Sequential()
# model.add(hub_layer)
# model.add(tf.keras.layers.Dense(16, activation = 'relu'))
# model.add(tf.keras.layers.Dense(1))
# model.summary()

#
# import numpy as np
# import tensorflow as tf
# import tensorflow_hub as hub
# import tensorflow_datasets as tfds
# from tensorflow_hub.keras_layer import KerasLayer
#
# from tensorflow import keras
# from tensorflow.keras import layers
# import tensorflow_hub as hub
#
#
# # Load the dataset
# train_data, validation_data, test_data = tfds.load(
#     name="imdb_reviews",
#     split=('train[:60%]', 'train[60%:]', 'test'),
#     as_supervised=True
# )
#
# # Preview batch
# train_example_batch, train_labeled_batch = next(iter(train_data.batch(10)))
# print(train_example_batch)
# print(train_labeled_batch)
#
# # Define embedding URL
# embedding_url = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
#
# # Preview output of embedding (optional)
# temp_layer = hub.KerasLayer(embedding_url, input_shape=[], dtype=tf.string, trainable=True)
# print(temp_layer(train_example_batch[:3]))  # Preview works
#
# # ✅ Create new embedding layer to be added to model
# hub_layer = hub.KerasLayer(embedding_url, input_shape=[], dtype=tf.string, trainable=True)
#
# # Define model using Sequential
# # model = tf.keras.Sequential([
# #     hub_layer,
# #     tf.keras.layers.Dense(16, activation='relu'),
# #     tf.keras.layers.Dense(1)
# # ])
# from tensorflow import keras
# from tensorflow.keras import layers
#
# model = keras.Sequential([
#     hub_layer,
#     layers.Dense(16, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])
#
#
# model.summary()





import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# Load and prepare the IMDB dataset
train_data, validation_data = tfds.load(
    name="imdb_reviews",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True
)

# Batch and prefetch for performance
train_data = train_data.shuffle(10000).batch(512).prefetch(tf.data.AUTOTUNE)
validation_data = validation_data.batch(512).prefetch(tf.data.AUTOTUNE)

# Define the input layer
input_text = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text_input")

# Load the Universal Sentence Encoder from TensorFlow Hub
embedding_layer = hub.KerasLayer(
    "https://tfhub.dev/google/nnlm-en-dim50/2",
    input_shape=[],
    dtype=tf.string,
    trainable=True,
    name="embedding"
)

# Use the embedding layer on the input
embedding = embedding_layer(input_text)

# Add a dense layer
dense = tf.keras.layers.Dense(16, activation='relu')(embedding)

# Final output layer (binary classification)
output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

# Define the model
model = tf.keras.Model(inputs=input_text, outputs=output)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data,
          validation_data=validation_data,
          epochs=5)









