# Import des librairies
import numpy as np
import pandas as p
import pickle
from bs4 import BeautifulSoup

# Tokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Nettoyage
def remove_code(html):
    # parse html content
    soup = BeautifulSoup(html, "html.parser")
    for data in soup(['style', 'script']):
        # Remove tags
        data.decompose()
        # return data by retrieving the tag content
    return soup.stripped_strings

def regex(text):
    # Remove html tags
    text = re.sub(r"<[^>]*>", ' ', text)
    # Remove usernames "@"
    text = re.sub(r'@\S+', ' ', text)
    # Remove hashtags
    text = re.sub(r'#\S+', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove unicode characters
    text = text.encode("ascii", "ignore").decode()
    # Remove irrelevant characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Remove English contractions
    text = re.sub("\'\w+", '', text)
    # Remove extra spaces
    text = re.sub('\s+', ' ', text)
    # Remove numbers
    text = re.sub(r'\w*\d+\w*', '', text)
    # Remove links
    text = re.sub(r'http*\S+', '', text)
    # Remove whitespace
    text = re.sub(r'^\s*|\s\s*', ' ', text).strip()
    text = text.replace('\n', '')

    return text


def remov_duplicates(input):
    # split input string separated by space
    input = input.split(" ")

    # now create dictionary using counter method
    # which will have strings as key and their
    # frequencies as value
    UniqW = Counter(input)

    # joins two adjacent elements in iterable way
    s = " ".join(UniqW.keys())

    return s

# Stop words
stop_w = list(set(stopwords.words('english'))) + ['[', ']', ',', '.', ':', '?', '(', ')']

def stop_word_filter_fct(list_words) :
    filtered_w = [w for w in list_words if not w in stop_w]
    filtered_w2 = [w for w in filtered_w if len(w) > 2]
    return filtered_w2

# lower case et alpha
def lower_start_fct(list_words) :
    lw = [w.lower() for w in list_words if (not w.startswith("@"))
                                       and (not w.startswith("#"))
                                       and (not w.startswith("http"))]
    return lw

# Remove single character
def remove_single_char_func(text, threshold=1):
    threshold = threshold
    words = word_tokenize(text)
    text = ''.join([w for w in words if len(w) > threshold])
    return text


