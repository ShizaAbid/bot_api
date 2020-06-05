import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from nltk.tokenize import word_tokenize
import gensim
import string
from spellchecker import SpellChecker

spell = spell = SpellChecker()
max_words = 20000
stemmer= SnowballStemmer("english")
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
stopwords = nltk.corpus.stopwords.words('english')
lemmer = nltk.stem.WordNetLemmatizer()

def spell_corr(query):
    tokens=[word.lower() for sent in nltk.sent_tokenize(query) for word in nltk.word_tokenize(sent)]
    corr_tokens = []
    for word in tokens:
        misspelled = spell.unknown([word])
        if(len(misspelled) == 0):
            corr_tokens.append(word)
        else:
            for words in misspelled:
                corr = spell.correction(words)
                corr_tokens.append(corr)
    print(corr_tokens)

    return  (" ".join(map(str,corr_tokens)))


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

def tok_behavior_flex(x):
    tok_module  = Tokenizer(num_words = 200,oov_token="<OOV>")
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

def model_score_LSTM_tokenize_stop(query,tokenizer,m):
    sen = tokenizing_stop(query)
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

def percentile_extract(query):
    percentile = ""
    tokens=[word.lower() for sent in nltk.sent_tokenize(query) for word in nltk.word_tokenize(sent)]
    print(tokens)
    for i in range(len(tokens)):
        if(re.search('\d+',tokens[i])):
            if(i+1 < len(tokens)):
                if(re.search('%|percent',tokens[i+1])):
                    percentile = tokens[i]
    return percentile

def percentile_change(query):
    entity = ""
    if(re.search('(((above \d+( |)(%|percent))|(\d+( |)(%|percent)( and| &| or)* (above|high|more)))|(((more|higher) (then|than) \d)))',query)):
        entity = ('"Percentile_Change":"Above"')
    elif(re.search('(((below \d+( |)(%|percent))|(\d+( |)(%|percent)( and| &| or)* (below| low| less)))|((( less| low) (then|than) \d)))',query)):
        entity = ('"Percentile_Change":"Below"')
    return entity


