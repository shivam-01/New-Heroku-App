from flask import Flask, render_template, request, redirect
import app_helper
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import base64

# __name__ == __main__
app = Flask(__name__)

# Model Config
max_length = 45
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
features_shape = 2048
attention_features_shape = 64

with open('tokenizer.pickle', 'rb') as temp:
    tokenizer = pickle.load(temp)

def decode_image(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

def evaluate(image, encoder, decoder, image_features_extract_model):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = tf.zeros((1, units))

    temp_input = tf.expand_dims(decode_image(image), 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def caption_it(b64_string, encoder, decoder, image_features_extract_model):
    image =  base64.b64decode(b64_string)
    result, attention_plot = evaluate(image, encoder, decoder, image_features_extract_model)
    return ' '.join(result[:-1])

# model directories
encoder_dir = 'encoder/'
decoder_dir = 'decoder/'

# loading models
encoder = tf.saved_model.load(encoder_dir)
decoder = tf.saved_model.load(decoder_dir)

# load the Inception v3 model
image_features_extract_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
    caption = ""
    if request.method == 'POST':
        file = request.files['userfile'].read()
        filebyte = app_helper.encode_to_base64(file)
        caption = caption_it(filebyte, encoder, decoder, image_features_extract_model)
        
    return render_template("index.html", image_caption = caption)

if __name__ == '__main__':
    app.run()
