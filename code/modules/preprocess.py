import torch
import pandas as pd
import nltk
import re
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('german'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenize the text
    if isinstance(text, list):
        text = ' '.join(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
    
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word.lower() not in stop_words]
    # Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)