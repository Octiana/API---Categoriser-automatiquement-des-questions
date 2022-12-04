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

def tokenizer_fct(sentence) :
    # print(sentence)
    sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ')
    word_tokens = word_tokenize(sentence_clean)
    return word_tokens
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

def lemma_fct(list_words) :
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]
    return lem_w

def transform_bow_fct(desc_text) :
    text_regex = regex(desc_text.lower())
    text_remove_duplicated = remov_duplicates(text_regex)
    word_tokens = tokenizer_fct(text_remove_duplicated)
    sw = stop_word_filter_fct(word_tokens)
    lem_w = lemma_fct(sw)
    transf_desc_text = ' '.join(lem_w)
    return transf_desc_text

#### Les modeles:

class LdaModel:

    def __init__(self):
        filename_model = "./models/lda_model.pkl"
        filename_dictionary = "./models/dictionary.pkl"
        self.model = pickle.load(open(filename_model, 'rb'))
        self.dictionary = pickle.load(open(filename_dictionary, 'rb'))

    def predict_tags(self, text):
        """
        Predict tags of a preprocessed text

        Args:
            text(list): preprocessed text
        Returns:
            res(list): list of tags
        """
        corpus_new = self.dictionary.doc2bow(text)
        topics = self.model.get_document_topics(corpus_new)

        # find most relevant topic according to probability
        relevant_topic = topics[0][0]
        relevant_topic_prob = topics[0][1]

        for i in range(len(topics)):
            if topics[i][1] > relevant_topic_prob:
                relevant_topic = topics[i][0]
                relevant_topic_prob = topics[i][1]

        # retrieve associated to topic tags present in submited text
        res = self.model.get_topic_terms(topicid=relevant_topic, topn=20)

        res = [self.dictionary[tag[0]] for tag in res if self.dictionary[tag[0]] in text]

        return res


class SupervisedModel:

    def __init__(self):
        filename_supervised_model = "./models/logit_model.pkl"
        filename_tfidf_model = "./models/tfidf_model.pkl"
        filename_mlb_model = "./models/mlb_model.pkl"

        self.supervised_model = pickle.load(open(filename_supervised_model, 'rb'))
        self.tfidf_model = pickle.load(open(filename_tfidf_model, 'rb'))
        self.mlb_model = pickle.load(open(filename_mlb_model, 'rb'))
    def predict_tags(self, text):
        """
        Predict tags according to a lemmatized text using a supervied model.

        Args:
            supervised_model(): Used mode to get prediction
            mlb_model(): Used model to detransform
        Returns:
            res(list): List of predicted tags
        """
        input_vector = self.tfidf_model.transform(text)
        input_vector = pd.DataFrame(input_vector.toarray())

        res = self.supervised_model.predict(input_vector)
        res = self.mlb_model.inverse_transform(res)
        res = list({tag for tag_list in res for tag in tag_list if (len(tag_list) != 0)})
        res = [tag for tag in res if tag in text]

        return res

