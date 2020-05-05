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
sys.path.insert(0,'D:\\bot\\botapi\\botapi')
from functions import token_stems
from functions import token_stems_stop
from functions import tok
from functions  import tokenizing
#from leave import leave_func
from main_model import mainmodel_func
from identification import identification
from functions import tok_behavior
from preprocessing import data_preprocessing

pre_process_data = data_preprocessing()


#API WILL RUN FROM HERE

def load_libraries(request):
    return HttpResponse("<h1>Not Much Going On Here 5</h1>")

def bot_API(request,name):
    #result = mainmodel_func(name,main_tok,module_tok)
    result = identification(name,pre_process_data[0],pre_process_data[1],pre_process_data[2],pre_process_data[3])
    return HttpResponse(result)


#hp = module_tok(help_desk_tok,help_desk_satisfication_tok,leave_tok)
#print(leave_func(query,hp.Leave()))
#print(helpdesk_func(query,hp.Helpdesk(),hp.Helpdesk_satisfication()))
