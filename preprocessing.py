import pandas as pd
import nltk
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
from functions import genism_model_creation
from functions import token_stems
from functions import token_stems_stop
from functions import tok
from functions  import tokenizing
from functions  import tok_behavior_flex
#from leave import leave_func
from identification import identification
from functions import tok_behavior
from functions import tokenizing_stop



Stopwords = nltk.corpus.stopwords.words('english')
#LOADING DATASET
#dataset = pd.read_excel('D:\\bot\\botapi\\botapi\\dataset\\Main.xlsx')
#data = [sent for sent in dataset['Data']]

def data_preprocessing():
    dataset = pd.read_excel('D:\\bot\\botapi\\botapi\\dataset\\Main_dataset(temp).xlsx')
    #HELP DESK TOKENIZER


    '''help_desk_data = data[data['Module'] == 'helpdesk']
    docs= help_desk_data['Data']
    x_helpdesk = []
    for i in docs:
        x_helpdesk.append(token_stems_stop(i))
    help_desk_tok = tok(x_helpdesk)
    #HELPDESK SATISFICATION
    help_desk_satisfication = help_desk_data[help_desk_data['sub_intent_one'] == 'satisfication_level']
    docs= help_desk_satisfication['Data']
    x_helpdesk_satisfication = []
    for i in docs:
        x_helpdesk_satisfication.append(token_stems_stop(i))
    help_desk_satisfication_tok = tok(x_helpdesk_satisfication)
    #LEAVE MODULE TOKEIZER
    leave_data = data[data['Module'] == 'Leave']
    docs = leave_data['Data']
    x_leave = []
    for i in docs:
        x_leave.append(token_stems_stop(i))
    leave_tok = tok(x_leave)'''



                                        ### PRE-PROCESSING STEPS FOR IDENTIFICATION ###
    sent_tokens = [sent for sent in dataset['Data']]


                                        ### TOKENIZATION FOR QUANTITY MODEL ###
    Entity = []
    docs = dataset['Data']
    x_quantitydata = []
    for i in docs:
        x_quantitydata.append(token_stems(i))
    quantity_tok = tok( x_quantitydata)
    Entity.append(quantity_tok)

                                        ### EMPLOYEE DETAIL ###

    name_database = pd.read_excel("D:\\bot\\botapi\\botapi\\dataset\\SoftronicEmployee.xlsx")
    emp_detail = []
    emp_detail.append(name_database['Name'])
    emp_detail.append(name_database['EmpCode'])
    emp_detail.append(name_database['EmpId'])
    Entity.append(emp_detail)

                                        ### TOKENIZING FOR MODELS ###
    #MAIN_DATASET_TOKENIZING
    docs = dataset['Data']
    x_maindata = []
    for i in docs:
        x_maindata.append(token_stems(i))
    main_tok = tok(x_maindata)

    module_tok = []

                                        ### LEAVE TOKENIZERS ###
    #LEAVE_DATASET_TOKENIZING
    leave_entities = []
    leave_dataset = dataset[dataset['Module'] == 'leave']
    x_leave = []
    docs = leave_dataset['Data']
    for i in docs:
        x_leave.append(token_stems(i))
    leave_main_tok = tok(x_leave)
    leave_entities.append(leave_main_tok)

    #EMOTIONAL_LEAVES
    #SICK
    sick = leave_dataset[leave_dataset['sub_intent_two'] == 'sick']
    sick_intent = genism_model_creation(sick['Data'])
    sims_s = gensim.similarities.Similarity('D:\\bot\\botapi\\botapi\\dataset\\Gensim_data\\sick.txt',sick_intent[1],len(sick_intent[2]))
    #CASUAL
    casual = leave_dataset[leave_dataset['sub_intent_two'] == 'casual']
    casual_intent = genism_model_creation(casual['Data'])
    sims_c = gensim.similarities.Similarity('D:\\bot\\botapi\\botapi\\dataset\\Gensim_data\\casual.txt',casual_intent[1],len(casual_intent[2]))
    emotional_leaves = {1:sims_s,2:sick_intent[3],3:sick_intent[2],4:sims_c,5:casual_intent[3],6:casual_intent[2]}
    leave_entities.append(emotional_leaves)

    #LEAVE_APPROVAL
    #APPROVE
    approve = leave_dataset[leave_dataset['sub_intent_two'] == 'approve']
    approve_intent = genism_model_creation(sick['Data'])
    sims_approve = gensim.similarities.Similarity('D:\\bot\\botapi\\botapi\\dataset\\Gensim_data\\approve.txt',approve_intent[1],len(approve_intent[2]))
    #REJECT
    reject = leave_dataset[leave_dataset['sub_intent_two'] == 'reject']
    reject_intent = genism_model_creation(reject['Data'])
    sims_reject = gensim.similarities.Similarity('D:\\bot\\botapi\\botapi\\dataset\\Gensim_data\\reject.txt',reject_intent[1],len(reject_intent[2]))
    leave_approval_gensim = {1:sims_approve,2:approve_intent[3],3:approve_intent[2],4:sims_reject,5:reject_intent[3],6:reject_intent[2]}
    leave_entities.append(leave_approval_gensim)

    module_tok.append(leave_entities)


                                        ### SEPARATION TOKENIZERS ###
    separation_entities = []
    separation_dataset = dataset[dataset['Module'] == 'separation']
    x_separation = []
    docs = separation_dataset['Data']
    for i in docs:
        x_leave.append(token_stems(i))
    separation_main_tok = tok(x_separation)
    separation_entities.append(separation_main_tok)
    module_tok.append(separation_entities)


                                        ### BEHAVIORAL TOKENIZERS ###
    behavioral_entities = []
    BA = dataset[dataset['Module'] == 'Behavioral_profile']
    x_BA = []
    docs = BA['Data']
    for i in docs:
        x_BA.append(tokenizing(i))
    BA_tok = tok_behavior(x_BA)
    behavioral_entities.append(BA_tok)


    #PERSONA TOKENIZERS

    BA_persona = BA[BA['sub_intent'] == 'Persona']
    x_BA_persona = []
    docs = BA_persona['Data']
    for i in docs:
        x_BA_persona.append(tokenizing(i))
    BA_persona_tok = tok(x_BA_persona)
    behavioral_entities.append(BA_persona_tok)


    #FLEX TOKENIZERS
    Flex_persona = BA[BA['sub_intent_two'] == 'flex']
    x_Flex = []
    docs = Flex_persona['Data']
    for i in docs:
        x_Flex.append(tokenizing(i))
    FLex_tok = tok_behavior_flex(x_Flex)
    behavioral_entities.append(FLex_tok)

    module_tok.append(behavioral_entities)

                                ### CAPABILITY TOKENIZERS ###

    capability_entities = []
    CA = dataset[dataset['Module'] == 'capability_profile']
    x_CA = []
    docs = CA['Data']
    for i in docs:
        x_CA.append(tokenizing(i))
    CA_tok = tok(x_CA)
    capability_entities.append(CA_tok)

    #SUB INTENT TOKENIZER
    CAA = CA[CA['sub_intent_two']!='overall']
    x_CA_sub_intent = []
    docs = CAA['Data']
    for i in docs:
        x_CA_sub_intent.append(tokenizing_stop(i))
    CA_sub_tok =tok(x_CA_sub_intent)
    capability_entities.append(CA_sub_tok)

    #OVERALL
    overall = CA[CA['sub_intent_two']=='overall']

    score = overall[overall['sub_intent_three'] == 'score']
    score_intent = genism_model_creation(score['Data'])
    sims_score = gensim.similarities.Similarity('D:\\bot\\botapi\\botapi\\dataset\\Gensim_data\\score.txt',score_intent[1],len(score_intent[2]))

    report = overall[overall['sub_intent_three'] == 'report']
    report_intent = genism_model_creation(report['Data'])
    sims_report = gensim.similarities.Similarity('D:\\bot\\botapi\\botapi\\dataset\\Gensim_data\\report.txt',report_intent[1],len(report_intent[2]))

    inquiry = overall[overall['sub_intent_three'] == 'inquiry']
    inquiry_intent = genism_model_creation(inquiry['Data'])
    sims_inquiry = gensim.similarities.Similarity('D:\\bot\\botapi\\botapi\\dataset\\Gensim_data\\inquiry.txt',inquiry_intent[1],len(inquiry_intent[2]))

    overall_gensim = {1:sims_score,2:score_intent[3],3:score_intent[2],4:sims_report,5:report_intent[3],6:report_intent[2],7:sims_inquiry,8:inquiry_intent[3],9:inquiry_intent[2]}
    capability_entities.append(overall_gensim)

    module_tok.append(capability_entities)

                                ### EMOTIONAL TOKENIZERS ###
    emotional_entities = []
    EA = dataset[dataset['Module'] == 'emotional_assessment']
    x_EA = []
    docs = EA['Data']
    for i in docs:
        x_EA.append(token_stems(i))
    EA_tok = tok_behavior_flex(x_EA)
    emotional_entities.append(EA_tok)


    #EMOTIONAL_MINDSET
    EM_EA = EA[EA['sub_intent_two'] == 'emotional_mindset']
    x_EM = []
    docs = EM_EA['Data']
    for i in docs:
        x_EM.append(tokenizing(i))
    tok_EM =  tok(x_EM)
    emotional_entities.append(tok_EM)

    #UNDERSTANDING_SELF_AND_OTHERS
    USO = EA[EA['sub_intent_two'] == 'Understanding Self & / and Others']
    x_USO = []
    docs = USO['Data']
    for i in docs:
        x_USO.append(tokenizing(i))
    tok_USO = tok(x_USO)
    emotional_entities.append(tok_USO)

    #MASTERING SELF WITH OTHERS
    MSO =  EA[EA['sub_intent_two'] == 'Mastering Self with others']
    x_MSO = []
    docs = MSO['Data']
    for i in docs:
        x_MSO.append(tokenizing(i))
    tok_MSO = tok(x_MSO)
    emotional_entities.append(tok_MSO)

    #MASTERING SELF
    MS =  EA[EA['sub_intent_two'] == 'Mastering Self']
    x_MS = []
    docs = MS['Data']
    for i in docs:
        x_MS.append(tokenizing(i))
    tok_MS = tok(x_MS)
    emotional_entities.append(tok_MS)

    #OVERALL
    overall = EA[EA['sub_intent_two']=='overall']

    score = overall[overall['sub_intent_three'] == 'overall_score']
    score_intent = genism_model_creation(score['Data'])
    sims_score = gensim.similarities.Similarity('D:\\bot\\botapi\\botapi\\dataset\\Gensim_data\\score.txt',score_intent[1],len(score_intent[2]))

    report = overall[overall['sub_intent_three'] == 'overall_report']
    report_intent = genism_model_creation(report['Data'])
    sims_report = gensim.similarities.Similarity('D:\\bot\\botapi\\botapi\\dataset\\Gensim_data\\report.txt',report_intent[1],len(report_intent[2]))

    inquiry = overall[overall['sub_intent_three'] == 'inquiry']
    inquiry_intent = genism_model_creation(inquiry['Data'])
    sims_inquiry = gensim.similarities.Similarity('D:\\bot\\botapi\\botapi\\dataset\\Gensim_data\\inquiry.txt',inquiry_intent[1],len(inquiry_intent[2]))

    overall_gensim = {1:sims_score,2:score_intent[3],3:score_intent[2],4:sims_report,5:report_intent[3],6:report_intent[2],7:sims_inquiry,8:inquiry_intent[3],9:inquiry_intent[2]}
    emotional_entities.append(overall_gensim)

    module_tok.append(emotional_entities)

    return(sent_tokens,Entity,main_tok,module_tok)
