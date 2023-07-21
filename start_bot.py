# %%
# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
import pickle
import pandas as pd
import numpy as np

import tensorflow.compat.v1 as tf

# %%
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
import time
import random


# Hotfix function
def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model


def make_keras_picklable():
    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__


# Run the function
make_keras_picklable()

# import our chat-bot intents file
import json

with open('intents.json') as json_data:
    intents = json.load(json_data)

data = pickle.load(open("katana-assistant-data.pkl", "rb"))
words = data['words']
classes = data['classes']
# %%
# create a data structure to hold user context
context = {}


# %%
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                # if show_details:
                # print ("found in bag: %s" % w)

    return np.array(bag)


# %%

# %%
# Use pickle to load in the pre-trained model
global graph
graph = tf.get_default_graph()

with open(f'katana-assistant-model.pkl', 'rb') as f:
    model = pickle.load(f)


# %%
def classify(sentence):
    ERROR_THRESHOLD = 0.25

    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    # filter out predictions below a threshold, and provide intent index
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
        # print(str(r[0]))
        # print(str(r[1]))
    # return tuple of intent and probability

    return return_list


def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # print(results[0][0])
                    answer = i['responses']
                    end = len(answer)
                    if end == 1:
                        print(answer[0])
                    elif end > 1:
                        print(answer[random.randint(0, end)])
                    else:
                        break            # if exception found just restart the session!

            results.pop(0)


# %%
while True:
    print('Ask your question please..')

    input1 = input()

    p = response(input1)

    print(p)
    time.sleep(3)
