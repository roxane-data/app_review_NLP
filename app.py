#===============================================================================================#

# Imports

#===============================================================================================#

import streamlit as st

import pandas as pd
import numpy as np

import pickle

from nltk.tokenize import RegexpTokenizer

from tensorflow.keras.models import load_model
from keras.preprocessing import sequence

#===============================================================================================#

# Functions and Models Prepared

#===============================================================================================#

word_index_dict = pickle.load(open('data/word_index_dict.pkl','rb'))

neural_net_model = load_model('data/models/Neural_Network.h5')

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

def index_review_words(text):
    review_word_list = []
    for word in text.lower().split():
        if word in word_index_dict.keys():
            review_word_list.append(word_index_dict[word])
        else:
            review_word_list.append(word_index_dict['<UNK>'])

    return review_word_list


def text_cleanup(text):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    token_list = tokenizer.tokenize(text.lower())
    new_text = ''
    for word in token_list:
        new_text += word + ' '

    return new_text
#===============================================================================================#

# Streamlit

#===============================================================================================#

st.title("Too good to go App Review Classifier")


review_text = st.text_area('Enter Your Review Here')

if st.button('Predict'):

    result_review = review_text.title()

    review_text = text_cleanup(review_text)

    review_text = index_review_words(review_text)

    review_text = sequence.pad_sequences([review_text],value=word_index_dict['<PAD>'],padding='post',maxlen=250)

    prediction = neural_net_model.predict(review_text)

    prediction = np.argmax(prediction)

    st.success(prediction+1)
