import tensorflow as tf
import numpy as np
import json
import sentencepiece as spm
from tensorflow.keras.layers import Embedding, GRU, Dense, LayerNormalization

#----------------------------------#
#       Tokenizer and Selector     #
#----------------------------------#

_sp = spm.SentencePieceProcessor()
_sp.load("subword.model")

def encode(text_to_encode):
    return _sp.encode(text_to_encode, out_type=int)

def decode(ids):
    return _sp.decode(ids)

def vocab_size():
    return _sp.get_piece_size()

def shape(ids_to_shape):
    while len(ids_to_shape) < 64:
        ids_to_shape.append(0)
    if len(ids_to_shape) > 64:
        ids_to_shape = ids_to_shape[-64:]
    return ids_to_shape

def sample_top_p(lgt, p=0.9, temperature=0.8):
    lgt = tf.cast(lgt, tf.float32) / temperature
    probs = tf.nn.softmax(lgt).numpy()

    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    cumulative = np.cumsum(sorted_probs)

    cutoff = cumulative <= p
    cutoff[np.argmax(cumulative > p)] = True

    filtered_indices = sorted_indices[cutoff]
    filtered_probs = sorted_probs[cutoff]
    filtered_probs /= filtered_probs.sum()

    return int(np.random.choice(filtered_indices, p=filtered_probs))

#------------------------#
#     File Management    #
#------------------------#

windows = np.load("dataset.npz")
X = windows["x_np"]
Y = windows["y_np"]

X = X.reshape((-1, 64))

def text_to_data():
    with open("training_data.txt", "r", encoding="utf-8") as r:
        text = r.read()
    data = encode(text)
    x_np = np.zeros((len(data) - 64, 64), dtype=np.int32)
    y_np = np.zeros((len(data) - 64,), dtype=np.int32)
    for j in range(len(data) - 64):
        x_np[j] = data[j:j + 64]
        y_np[j] = data[j + 64]
    np.savez_compressed("dataset.npz", x_np=x_np, y_np=y_np)
#------------------#
#    Main Model    #
#------------------#

aurora = tf.keras.Sequential([
    Embedding(input_dim=vocab_size()+1, output_dim=192, input_length=64),
    GRU(256, return_sequences=True, dropout=0.1, recurrent_dropout=0.1),
    LayerNormalization(),
    GRU(256, dropout=0.1, recurrent_dropout=0.1),
    LayerNormalization(),
    Dense(vocab_size()+1),
])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=len(X) // 64,
    decay_rate=0.95,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
aurora.build((None, 64))

#-------------------#
#   Training Loop   #
#-------------------#

def training_loop(epochs):
    with (tf.device("/GPU:0")):
        step = 0
        for j in range(epochs):
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            dataset = (tf.data.Dataset.from_tensor_slices((X_shuffled, Y_shuffled)).batch(64, drop_remainder=True))
            for x_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    aurora_prediction = aurora(x_batch, training=True)
                    loss = loss_fn(y_batch, aurora_prediction)
                gradients = tape.gradient(loss, aurora.trainable_variables)
                optimizer.apply_gradients(zip(gradients, aurora.trainable_variables))
                step += 1
                print(f"Epoch {j+1} progress: {(step / len(dataset)) * 100}%")


#--------------------------#
#  Save / Load Parameters  #
#--------------------------#

def save_params(model):
    params = []
    for layer in model.layers:
        weights = layer.get_weights()
        weights_list = [w.tolist() for w in weights]
        params.append(weights_list)

    with open("params.json", "w") as g:
        json.dump(params, g)

def load_params(model):
    with open("params.json", "r") as h:
        params = json.load(h)

    for layer, layer_weights in zip(model.layers, params):
        np_weights = [np.array(w, dtype=np.float32) for w in layer_weights]
        layer.set_weights(np_weights)

#--------------#
#     Main     #
#--------------#

print("would you like to test or train the model?")
user_input = input("> ")
if user_input == "test":
    load_params(aurora)
    text_input = input("> ")
    test_input = shape(encode(text_input))
    test_input = np.array(test_input, dtype=np.int32)
    test_input = np.expand_dims(test_input, axis=0)
    print(text_input, end="")
    for i in range(64):
        logits = aurora(test_input)[0]
        next_token = sample_top_p(logits)
        output = decode(next_token)
        print(output, end="")
        test_input = text_input + output
        test_input = shape(encode(test_input))
        test_input = np.array(test_input, dtype=np.int32)
        test_input = np.expand_dims(test_input, axis=0)

if user_input == "train":
    TEST_TRAIN = False
    if TEST_TRAIN:
        X = X[:2000]
        Y = Y[:2000]
    load_params(aurora)
    training_loop(epochs=1)
    if not TEST_TRAIN:
        save_params(aurora)