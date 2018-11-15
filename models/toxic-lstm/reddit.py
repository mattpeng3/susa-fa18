import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from util.tokenizer_helpers import *
tqdm.pandas()

# Basic cleaning
from html.parser import HTMLParser
class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

# Rebuild saved tokenizer
tokenizer = load_tokenizer('save/tokenizer.pickle')

# Load the test data
test_data = pd.read_csv('../../data/reddit-selfposts/rspct.tsv', sep='\t')

test_sent = test_data['selftext'].progress_apply(strip_tags) # Clean the data
test_tokens = tokenizer.texts_to_sequences(test_sent) # Convert to sequences
test = pad_sequences(test_tokens, maxlen=300) # Fixed-length sequences

model = load_model('save/model.h5')

submission = pd.read_csv('../../data/reddit-selfposts/rspct.tsv', sep='\t')
y_pred = model.predict(test, batch_size=1024, verbose=1)
submission["toxic"] = y_pred[:,0]
submission["severe_toxic"] = y_pred[:,1]
submission["obscene"] = y_pred[:,2]
submission["threat"] = y_pred[:,3]
submission["insult"] = y_pred[:,4]
submission["identity_hate"] = y_pred[:,5]
submission.to_csv('../../data/reddit-analyze.csv', index=False)
