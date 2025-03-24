import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.layers import LSTM,Dense,Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
import random
import numpy as np
import streamlit as st
from utils import sample, generate_text

filepath = tf.keras.utils.get_file('shakespeare.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
#extract the text, lower to improve accuracy
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
#training set
text = text[100000:800000]
charecters = sorted(set(text))

#text to numeric and back to text
char_to_index = dict((c,i) for i,c in enumerate(charecters))
index_to_char = dict((i,c) for i,c in enumerate(charecters))

#sentence passed to nueral network will generate/predict next charecters
SEQ_LENGTH = 45
STEP_SIZE = 3

# code used to create model
# sentences = []
# next_chars = []

# #filling up the list based on step size and seq size
# for i in range(0,len(text)-SEQ_LENGTH,STEP_SIZE):
#     sentences.append(text[i:i+SEQ_LENGTH])
#     next_chars.append(text[i+SEQ_LENGTH])

# #3D array
# x = np.zeros((len(sentences), SEQ_LENGTH, len(charecters)), dtype=np.bool)
# y = np.zeros((len(sentences), len(charecters)),dtype = np.bool)
 
# for i,sentence in enumerate(sentences):
#     for j,charecter in enumerate(sentence):
#         x[i,j,char_to_index[charecter]] = 1
#     y[i,char_to_index[next_chars[i]]] = 1


# #nueral network
# model = Sequential()
# model.add(LSTM(128,input_shape = (SEQ_LENGTH,len(charecters))))
# model.add(Dense(len(charecters)))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy',optimizer = RMSprop(learning_rate=0.01))

# model.fit(x,y,batch_size = 256,epochs = 4)
# model.save('textgenerator.keras')

#Loading a saved model
model = tf.keras.models.load_model('textgenerator.keras')

# Streamlit app
st.markdown(
    """
    <style>
    body {
        background-color: #f2f5f9;
        font-family: Arial, sans-serif;
    }

    .main-header {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 25px;
    }

    .sidebar .sidebar-content {
        background-color: #eef1f5;
    }

    .stButton > button {
        background-color: #1f77b4;
        color: white;
        font-size: 16px;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }

    .stButton > button:hover {
        background-color: #1a5e8e;
    }

    .stSlider {
        color: #1f77b4;
    }

    textarea {
        font-size: 14px;
        line-height: 1.6;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='main-header'>AI Text Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>A character-level text generator trained on Shakespeare!</p>", unsafe_allow_html=True)

# Input options
length = st.slider("Output Length (characters):", min_value=100, max_value=1000, step=50, value=300)
temperature = st.slider("Temperature (creativity):", min_value=0.1, max_value=1.0, step=0.1, value=0.5)

if st.button("Generate Text"):
    with st.spinner("Generating text..."):
        generated_text = generate_text(model, text, length, temperature, SEQ_LENGTH, char_to_index, index_to_char)
    st.text_area("Generated Text:", generated_text, height=300)



