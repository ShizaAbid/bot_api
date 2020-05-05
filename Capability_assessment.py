from keras.models import load_model
from os import path
import sys
#from help_desk_func import token_stems
from functions import model_score_LSTM_tokenize
from leave import leave_func
from functions import model_score_LSTM_behavior
#from help_desk_func import model_score_LSTM
import joblib
import re

def capability_assessment_model(query,capability_entity,entity):
    top_intent = ''
    Score = ''
    m = load_model('D:\\bot\\botapi\\botapi\\models\\Behavioral_Assesment\\Behavioral_model.h5py')
    score = model_score_LSTM_tokenize(query,capability_entity[0],m)
    if((score[0][0]>score[0][1])&(score[0][0]>score[0][2])&(score[0][0]>score[0][3])&(score[0][0]>score[0][4])&(score[0][0]>score[0][5])):
        top_intent = 'Verbal Interpretation'
    elif((score[0][1]>score[0][0])&(score[0][1]>score[0][2])&(score[0][1]>score[0][3])&(score[0][1]>score[0][4])&(score[0][1]>score[0][5])):
        top_intent = 'numerical_reasoning'
    elif((score[0][2]>score[0][0])&(score[0][2]>score[0][3])&(score[0][2]>score[0][3])&(score[0][2]>score[0][4])&(score[0][2]>score[0][5])):
        top_intent = 'spatial Visualization '
    elif((score[0][3]>score[0][0])&(score[0][3]>score[0][2])&(score[0][3]>score[0][1])&(score[0][3]>score[0][4])&(score[0][3]>score[0][5])):
        top_intent = 'overall'
    elif((score[0][4]>score[0][0])&(score[0][4]>score[0][2])&(score[0][4]>score[0][3])&(score[0][4]>score[0][1])&(score[0][4]>score[0][5])):
       top_intent = 'Perceptual Speed'
    else:
        top_intent = 'verbal_reasoning'
    Score = str(Score)
    return ('{"TopIntent": "'+top_intent+'", "Percentage":'+Score,entity)
