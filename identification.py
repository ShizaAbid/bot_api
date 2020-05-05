from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from functions import LemNormalize
import re
from langdetect import detect
from main_model import mainmodel_func
import string
from entities import entity_extract

TfidfVec = TfidfVectorizer(tokenizer=LemNormalize,max_df = 650,stop_words = 'english')


def identification(user_response,sent_tokens,entity_functions,main_tok,module_tok):
    entity = entity_extract(user_response,entity_functions[0],entity_functions[1])
    #entit =',"Entities": {' + ",".join(map(str,entity))+'}'
    print(entit)
    name = user_response
    sent_tokens.append(user_response)
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    tokens=[word.lower() for sent in nltk.sent_tokenize(user_response) for word in nltk.word_tokenize(sent)]
    if((req_tfidf==0)|(len(tokens)<3)):
        #tokens=[word.lower() for sent in nltk.sent_tokenize(user_response) for word in nltk.word_tokenize(sent)]
        text = " ".join(map(str,tokens))
        if(re.search('hi|hello|how are you',text)):
            Top_intent = '{"TopIntent": "Greeting"'
        elif(len(user_response)<3):
            dec = detect(user_response)
            entit =',"Entities": {' + ",".join(map(str,entity))+'}'
            percentage = '"Percentage":100.0'
            if(dec == 'en'):
                response = ('{"TopIntent": "None",' + percentage + entit + '}')
            else:
                response = ('{"TopIntent": "NoneLang",' + percentage + entit + '}')
        else:
            Top_intent = '{"TopIntent": "None"'
        entit = Top_intent +',"Entities": {' + ",".join(map(str,entity))+'}'
        response = entit+'}'


    else:
        response = mainmodel_func(name,main_tok,module_tok,entity)
        entit =',"Entities": {' + ",".join(map(str,response[1]))+'}'
        response = response[0]+entit+'}'
    return(response)
