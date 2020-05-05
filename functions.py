import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from nltk.tokenize import word_tokenize
import gensim
import string

max_words = 20000
stemmer= SnowballStemmer("english")
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
stopwords = nltk.corpus.stopwords.words('english')
lemmer = nltk.stem.WordNetLemmatizer()



def stemming(text):
    stems =[stemmer.stem(t) for t in text]
    return stems

def token_stems(text):
    tokens=tokenizing(text)
    stems=stemming(tokens)
    return stems

def tokenizing(text):
    tokens=[word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens=[]
    for token in tokens:
        if (re.search('[a-zA-Z]|\'', token)):
            filtered_tokens.append(token)
    return filtered_tokens

def token_stems_stop(text):
    tokens=tokenizing_stop(text)
    stems=stemming(tokens)
    return stems

def tokenizing_stop(text):
    tokens=[word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens=[]
    for token in tokens:
        if (re.search('[a-zA-Z]|\'', token)):
            if token not in stopwords:
                filtered_tokens.append(token)
    return filtered_tokens

def tok(x):
    tok_module  = Tokenizer(num_words = max_words)
    tok_module.fit_on_texts(x)
    return(tok_module)

def tok_behavior(x):
    tok_module  = Tokenizer(num_words = 200)
    tok_module.fit_on_texts(x)
    return(tok_module)

def model_score_LSTM(query,tokenizer,m):
    sen = token_stems(query)
    sen_test = ([list(sen)])
    print(sen_test)
    sen_sequences = tokenizer.texts_to_sequences(sen_test)
    sen_sequences_matrix = sequence.pad_sequences(sen_sequences,maxlen = 200)
    score = m.predict(sen_sequences_matrix)
    #print(score)
    return(score)

def model_score_LSTM_stop(query,tokenizer,m):
    sen = token_stems_stop(query)
    sen_test = ([list(sen)])
    print(sen_test)
    sen_sequences = tokenizer.texts_to_sequences(sen_test)
    sen_sequences_matrix = sequence.pad_sequences(sen_sequences,maxlen = 200)
    score = m.predict(sen_sequences_matrix)
    #print(score)
    return(score)

def model_score_LSTM_tokenize(query,tokenizer,m):
    sen = tokenizing(query)
    sen_test = ([list(sen)])
    print(sen_test)
    sen_sequences = tokenizer.texts_to_sequences(sen_test)
    sen_sequences_matrix = sequence.pad_sequences(sen_sequences,maxlen = 200)
    score = m.predict(sen_sequences_matrix)
    #print(score)
    return(score)

def model_score_LSTM_behavior(query,tokenizer,m):
    sen = tokenizing(query)
    sen_test = ([list(sen)])
    print(sen_test)
    sen_sequences = tokenizer.texts_to_sequences(sen_test)
    sen_sequences_matrix = sequence.pad_sequences(sen_sequences,maxlen = 2000)
    score = m.predict(sen_sequences_matrix)
    #print(score)
    return(score)


def genism_model_creation(data):
    gen_doc = [[w.lower() for w in word_tokenize(text)]for text in data]
    dictionary = gensim.corpora.Dictionary(gen_doc)
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_doc]
    tf_idf = gensim.models.TfidfModel(corpus)
    return {1:tf_idf[corpus],2:dictionary,3:tf_idf}

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


