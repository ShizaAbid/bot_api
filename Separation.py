from keras.models import load_model
from os import path
import sys
from functions import model_score_LSTM
from nltk.tokenize import word_tokenize
import numpy as np
import re

def separation_func(query,separation_entities,entities):
    sub_intent = ''
    query = query.lower()
    m = load_model("D:\\bot\\botapi\\botapi\\models\\Separation\\Sepration.h5")
    score = model_score_LSTM(query,separation_entities[0],m)
    if((score[0][0]>score[0][1])&(score[0][0]>score[0][2])&(score[0][0]>score[0][3])&(score[0][0]>score[0][4])&(score[0][0]>score[0][5])):
        sub_intent = "Blacklist Employees"
        Score = round(score[0][0]*100,2)
    elif((score[0][1]>score[0][0])&(score[0][1]>score[0][2])&(score[0][1]>score[0][3])&(score[0][1]>score[0][4])&(score[0][1]>score[0][5])):
        sub_intent = "Exit Clearance "
        #status = exit_clearance_module(name)
        #entities.append('"Status:""'+status+'"')
        Score = round(score[0][1]*100,2)
    elif((score[0][2]>score[0][0])&(score[0][2]>score[0][1])&(score[0][2]>score[0][3])&(score[0][2]>score[0][4])&(score[0][2]>score[0][5])):
        sub_intent = "Exit interview "
        #status = exit_interview_module(name)
        #entities.append('"Status:""'+status+'"')
        Score = round(score[0][2]*100,2)
    elif((score[0][3]>score[0][0])&(score[0][3]>score[0][1])&(score[0][3]>score[0][2])&(score[0][3]>score[0][4])&(score[0][3]>score[0][5])):
        sub_intent = "Exit Survey " #+
        #entities.append('"Status:""'+status+'"')
        #status = exit_survey_module(name)
        Score = round(score[0][3]*100,2)
    elif((score[0][4]>score[0][0])&(score[0][4]>score[0][1])&(score[0][4]>score[0][2])&(score[0][4]>score[0][3])&(score[0][4]>score[0][5])):
        sub_intent = "Notice Period "
        #entities.append('"Status:""'+status+'"')
        #status = notice_period_module(name)
        Score = round(score[0][4]*100,2)
    else:
        sub_intent = "Resignation" #+ resignation(name)
        Score = round(score[0][5]*100,2)
    Score = str(Score)
    return ('{"TopIntent": "'+ sub_intent +'", "Percentage":'+ Score,entities)