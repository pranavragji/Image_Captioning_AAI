import os
import io
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

seed = 111
np.random.seed(seed)
tf.random.set_seed(seed)

BASE_DIR = './input/flickr8k'
WORKING_DIR = './working'

vgg_model = VGG16()
# Restructure the model
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

models = {}

def loadAllModels():
    for m in getModelList():
        if m not in models:
            models[m] = load_model(os.path.join(WORKING_DIR,m))

def getModelList():
    directory_path = os.path.join(WORKING_DIR)
    file_list = [f for f in os.listdir(directory_path) if f.endswith(".h5")]
    return file_list

loadAllModels()

def createCaptionMap(filename):
    caption_mapping = {}
    text_data = []
    images_to_skip = set()
    
    with open(filename, 'r') as f:
        next(f)
        captions_doc = f.read()

        for line in tqdm(captions_doc.split('\n')):
            lineSplit = line.split(',')
            img_name, caption = lineSplit[0], lineSplit[1:]
            image_id = img_name.split('.')[0]
            caption = " ".join(caption)
            tokens = caption.strip().split()
            
            if image_id not in images_to_skip: 
                # We will add a start and an end token to each caption
                caption = "startseq " + caption.strip() + " endseq"
                text_data.append(caption)

                if image_id in caption_mapping:
                    caption_mapping[image_id].append(caption)
                else:
                    caption_mapping[image_id] = [caption]
        
        f.close()
    return caption_mapping, text_data

# Load the dataset
captions_mapping, text_data = createCaptionMap(os.path.join(BASE_DIR, 'captions.txt'))

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
vocab_size = len(tokenizer.word_index) + 1

max_length = max(len(caption.split()) for caption in text_data)

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # Add start tag for generation process
    in_text = 'startseq'
    # Iterate over the max length of sequence
    for i in range(max_length):
        # Encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # Predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # Get index with high probability
        yhat = np.argmax(yhat)
        # Convert index to word
        word = idx_to_word(yhat, tokenizer)
        # Stop if word not found
        if word is None:
            break
        
        in_text += " " + word
            
        if word == 'endseq':
            break
    return in_text

def getGeneratedCaption(uploaded_file, modelName, datType="FILE"):
    if datType == "FILE":
        image = Image.open(uploaded_file)
    else:
        image = Image.open(io.BytesIO(uploaded_file))
    image = image.resize((224,224))    
    image = img_to_array(image)    
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))    
    image = preprocess_input(image)    
    feature = vgg_model.predict(image, verbose=0)

    # Predict from the trained model
    return predict_caption(models[modelName], feature, tokenizer, 35)