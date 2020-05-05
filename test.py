


                                                                ###### LOADING ALL LIBRARIES #######


from django.http import HttpResponse
import tensorflow as tf
from sklearn.externals import joblib
import json
import xlrd
import traceback
import numpy as np #use to handle numeric data
import nltk #for nlp purpose
import pandas as pd #use for file that we read
import re #to handle regular expression
from keras.models import load_model #To load model
from keras import backend as K #to load the backend library that we are using
from tensorflow.keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from langdetect import detect
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import h5py
from datetime import datetime
from datetime import timedelta
import datefinder
from dateparser.search import search_dates
import spacy
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import gensim
from nltk.stem.snowball import SnowballStemmer
import sys
#sys.path.append('D:\\bot_Django\\virtual\\virtual_bot\\bot\\')
#from anc import check
import os
from os import path
from . import leave
from . import entities
#from . import personal
import pyodbc
from langdetect import detect
from functions import tok_behavior


emp_detail = []                                                 #################CONNECTING DATABASE######
emp_name = []
emp_code = []
emp_id = []
server = 'DBSRV2\SQL2017'
database = 'PeoplePartners'
username = 'developer'
password = '12hcms34%'
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()
cursor.execute("select EmpId,EmpCode,NameOnly as [Name],ReportTo as [ReportToId],tblReportToName as [ReportToName] from dbo.fn_EmployeeProfile('en-GB',-1);")
row = cursor.fetchall()
for entity in row:
    emp_name.append(entity.Name)
    emp_code.append(entity.EmpCode)
    emp_id.append(entity.EmpId)

emp_detail.append(emp_name)
emp_detail.append(emp_code)
emp_detail.append(emp_id)




                                                                ###### IDENTIFYING STATIC VARIABLES #######



stemmer= SnowballStemmer("english")
nlp = spacy.load('en_core_web_sm')
stopwords = nltk.corpus.stopwords.words('english')
max_len=200
max_words = 20000
tok = Tokenizer(num_words=max_words)
lemmer = nltk.stem.WordNetLemmatizer()
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)





                                                                    ###### IDENTIFYING FUNCTIONS #######



#LEMMATIZATION CODE
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

TfidfVec = TfidfVectorizer(tokenizer=LemNormalize,max_df = 650,stop_words = 'english')


#TOKENIZATION
def stemming(text):
    stems =[stemmer.stem(t) for t in text]
    return stems
def token_stems(text):
    tokens=tokenizing(text)
    stems=stemming(tokens)
    return stems
def tokenizing(text):
    #breaking each word and making them tokens
    tokens=[word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    #storing only alpha tokens
    filtered_tokens=[]
    for token in tokens:
        if (re.search('[a-zA-Z]|\'', token)):
            filtered_tokens.append(token)
    return filtered_tokens



                                                                ####### PRE-PROCESSING STEP FOR IDENTIFICATION ######
main_data = pd.read_excel('D:\\bot_Django\\virtual\\virtual_bot\\bot\\datasets\\test.xlsx')
data = main_data['Data']
sent_tokens = [sent for sent in data]
TfidfVec = TfidfVectorizer(tokenizer=LemNormalize,max_df = 250,stop_words = 'english')









                                                                ####### ALL THE PRE-PROCESSING STEPS FOR FILES ######


#sys.path.append('D:\\bot_Django\\virtual\\virtual_bot\\bot\\datasets\\')


                                                                ####### DATA LOADING FOR NAME ENTITY ######

'''name_database = pd.read_excel("D:\\bot_Django\\virtual\\virtual_bot\\bot\\SoftronicEmployee.xlsx")
emp_detail = []
emp_detail.append(name_database['Name'])
emp_detail.append(name_database['EmpCode'])
emp_detail.append(name_database['EmpId']) '''




                                                                ####### PRE-PROCESSING STEP FOR QUANTITY MODEL ######

data_quantity = pd.read_excel('D:\\bot_Django\\virtual\\virtual_bot\\bot\\datasets\\data.xlsx')
docs= data_quantity['Data']
tokens = []
for i in docs:
    temp = token_stems(i)
    tokens.append(temp)
x, y = np.asarray(tokens) , np.asarray(data_quantity['quantity'])
xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size= 0.3, random_state=100)
tok_quantity = Tokenizer(num_words=max_words)
tok_quantity.fit_on_texts(x)






                                                                ####### PRE-PROCESSING STEP FOR LEAVE MODEL ######

data_leave = pd.read_excel('D:\\bot_Django\\virtual\\virtual_bot\\bot\\datasets\\leave_final.xlsx')
docs= data_leave['Data']
tokens = []
for i in docs:
    temp = token_stems(i)
    tokens.append(temp)
x, y = np.asarray(tokens) , np.asarray(data_leave['Type'])
xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size= 0.3, random_state=100)
tok_leave = Tokenizer(num_words=max_words)
tok_leave.fit_on_texts(x)
Leave_entities = []





                                            ####### PRE-PROCESSING STEP FOR EMOTIONAL LEAVE RECOGNIATION ######



sick = data_leave[data_leave["sub_type_two"] == 'sick']
casual = data_leave[data_leave["sub_type_two"] == 'casual']
gen_docs_s = [[w.lower() for w in word_tokenize(text)]
            for text in sick['Data']]
gen_docs_c = [[w.lower() for w in word_tokenize(text)]
            for text in casual['Data']]
dictionary_s = gensim.corpora.Dictionary(gen_docs_s)
dictionary_c = gensim.corpora.Dictionary(gen_docs_c)
corpus_s = [dictionary_s.doc2bow(gen_doc_s) for gen_doc_s in gen_docs_s]
corpus_c = [dictionary_c.doc2bow(gen_doc_c) for gen_doc_c in gen_docs_c]
tf_idf_s = gensim.models.TfidfModel(corpus_s)
tf_idf_c = gensim.models.TfidfModel(corpus_c)
sims_s = gensim.similarities.Similarity('D:\\bot_Django\\virtual\\virtual_bot\\bot\\Gensim_data\\sick.txt',tf_idf_s[corpus_s],
                                num_features=len(dictionary_s))
sims_c = gensim.similarities.Similarity('D:\\bot_Django\\virtual\\virtual_bot\\bot\\Gensim_data\\casual.txt',tf_idf_c[corpus_c],
                                          num_features=len(dictionary_c))
leave_emotional_genism = {1:sims_s,2:tf_idf_s,3:dictionary_s,4:sims_c,5:tf_idf_c,6:dictionary_c}

Leave_entities.append(leave_emotional_genism)



                                            ####### PRE-PROCESSING STEP FOR LEAVE APPROVAL RECOGNIATION ######


approve = data_leave[data_leave['sub_type_two'] == 'approve']
reject = data_leave[data_leave['sub_type_two'] == 'reject']
gen_approve = [[w.lower() for w in word_tokenize(text)]
            for text in approve['Data']]
gen_reject = [[w.lower() for w in word_tokenize(text)]
            for text in reject['Data']]
dictionary_approve = gensim.corpora.Dictionary(gen_approve)
dictionary_reject = gensim.corpora.Dictionary(gen_reject)
corpus_approve = [dictionary_approve.doc2bow(gen_approve) for gen_approve in gen_approve]
corpus_reject = [dictionary_reject.doc2bow(gen_reject) for gen_reject in gen_reject]
tf_idf_approve  = gensim.models.TfidfModel(corpus_approve)
tf_idf_reject  = gensim.models.TfidfModel(corpus_reject)
sims_approve = gensim.similarities.Similarity('D:\\bot_Django\\virtual\\virtual_bot\\bot\\Gensim_data\\approve.txt',tf_idf_approve[corpus_approve],
                num_features=len(dictionary_approve))
sims_reject = gensim.similarities.Similarity('D:\\bot_Django\\virtual\\virtual_bot\\bot\\Gensim_data\\reject.txt',tf_idf_reject[corpus_reject],
                num_features=len(dictionary_reject))
leave_approval_gensim = {1:sims_approve,2:tf_idf_approve,3:dictionary_approve,4:sims_reject,5:tf_idf_reject,6:dictionary_reject}
Leave_entities.append(leave_approval_gensim)




                                            ####### PRE-PROCESSING STEP FOR LEAVE INQUIRY MODEL ######


data_leave_inquiry = data_leave[data_leave['Type'] == 'leave_inquiry']
docs= data_leave_inquiry['Data']
tokens = []
for i in docs:
    temp = token_stems(i)
    tokens.append(temp)
x, y = np.asarray(tokens) , np.asarray(data_leave_inquiry['sub_type_two'])
xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size= 0.3, random_state=100)
tok_leave_inquiry = Tokenizer(num_words=max_words)
tok_leave_inquiry.fit_on_texts(x)
Leave_entities.append(tok_leave_inquiry)
Leave_Inquiry_entities = []





                                            ####### PRE-PROCESSING STEP FOR LEAVE INQUIRY ENCASHMENT ######


leave_encash = data_leave[data_leave['sub_type_two'] == 'encashment']
docs= leave_encash['Data']
tokens = []
for i in docs:
    temp = token_stems(i)
    tokens.append(temp)
x, y = np.asarray(tokens) , np.asarray(leave_encash['TypeLeave'])
xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size= 0.3, random_state=100)
tok_leave_encash = Tokenizer(num_words=max_words)
tok_leave_encash.fit_on_texts(x)
Leave_Inquiry_entities.append(tok_leave_encash)

                                            ####### PRE-PROCESSING STEP FOR LEAVE INQUIRY SPECIFIC ######


leave_specific = data_leave[data_leave['sub_type_two'] == 'specific']
docs= leave_specific['Data']
tokens = []
for i in docs:
    temp = token_stems(i)
    tokens.append(temp)
x, y = np.asarray(tokens) , np.asarray(leave_specific['TypeLeave'])
xtrain, xtest, ytrain, ytest= train_test_split(x, y, test_size= 0.3, random_state=100)
tok_leave_specific = Tokenizer(num_words=max_words)
tok_leave_specific.fit_on_texts(x)
Leave_Inquiry_entities.append(tok_leave_specific)
Leave_entities.append(Leave_Inquiry_entities)




def load_libraries(request):
    return HttpResponse("<h1>Not Much Going On Here 5</h1>")

def hello(request,name):
    query = str(name)

    #WORK FOR LOG FILE
    curr_date = datetime.now()
    year = curr_date.strftime("%Y")
    month = curr_date.strftime("%B")
    file = 'D:\\bot_Django\\virtual\\virtual_bot\\bot\\log_file\\' + (str(year))
    if((path.exists(file)) == False):
        os.mkdir(file)

    file = file + "\\" + (str(month))+".txt"
    log_file = open(file, "a", encoding="utf")
    entity = entities.entity_extract(query,tok_quantity,emp_detail)
    entity = list(filter(None,entity))


    def identification(user_response,sent_tokens,entity):
        sent_tokens.append(user_response)
        tfidf = TfidfVec.fit_transform(sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        if(req_tfidf==0):
            tokens=[word.lower() for sent in nltk.sent_tokenize(user_response) for word in nltk.word_tokenize(sent)]
            text = " ".join(map(str,tokens))
            if(re.search('hi|hello|how are you',text)):
                Top_intent = '{"TopIntent": "Greeting"'
            elif(len(name)<3):
                dec = detect(name)
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
            response = leave.leave_module(query,tok_leave,entity,Leave_entities)
            entit =',"Entities": {' + ",".join(map(str,response[1]))+'}'
            response = response[0]+entit+'}'
        return(response)


    response = identification(name,sent_tokens,entity)
    if(query != 'favicon.ico'):
        log_file.write(query + '-->' + response +'\n')
    log_file.close()
    return HttpResponse(response)
    #response = leave.leave_module(query,tok_leave,entity,Leave_entities)
    #entity =',"Entities": {' + ",".join(map(str,response[1]))+'}'
    #response = response[0]+entity+'}'
    #print(entity)

    #name = ''

    #return HttpResponse(response)
    #name = ''







