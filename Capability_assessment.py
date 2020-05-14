from keras.models import load_model
from os import path
import sys
#from help_desk_func import token_stems
from functions import model_score_LSTM_tokenize
from leave import leave_func
from functions import model_score_LSTM_behavior
from functions import percentile_extract
#from help_desk_func import model_score_LSTM
from functions import model_score_LSTM_tokenize_stop
import joblib
import re

def capability_assessment_model(query,capability_entity,entity):
    top_intent = '"Assessment"'
    sub_intent = '"Capability_Assessment"'
    Score = ''
    if(re.search('verbal interpretation',query)):
        entity.append('"status" : "Verbal_Interpretation"')
        entity =  main_entity(query,capability_entity[1],entity)
    elif(re.search('numerical reasoning',query)):
        entity.append('"status" : "Numerical_Reasoning"')
        entity =  main_entity(query,capability_entity[1],entity)
    elif(re.search('spatial visualization',query)):
        entity.append('"status" : "Spatial_Visualization"')
        entity =  main_entity(query,capability_entity[1],entity)
    elif(re.search('perceptual speed',query)):
        entity.append('"status" : "Perceptual_Speed"')
        entity =  main_entity(query,capability_entity[1],entity)
    elif(re.search('verbal reasoning',query)):
        entity.append('"status" : "Verbal_Reasoning"')
        entity =  main_entity(query,capability_entity[1],entity)
    else:
        m = load_model('D:\\bot\\botapi\\botapi\\models\\Capability_Assessment\\capability_assessment.h5py')
        score = model_score_LSTM_tokenize(query,capability_entity[0],m)
        if((score[0][0]>score[0][1])&(score[0][0]>score[0][2])&(score[0][0]>score[0][3])&(score[0][0]>score[0][4])&(score[0][0]>score[0][5])):
            entity.append('"status" : "Verbal_Interpretation"')
            entity =  main_entity(query,capability_entity[1],entity)
        elif((score[0][1]>score[0][0])&(score[0][1]>score[0][2])&(score[0][1]>score[0][3])&(score[0][1]>score[0][4])&(score[0][1]>score[0][5])):
            entity.append('"status" : "Numerical_Reasoning"')
            entity =  main_entity(query,capability_entity[1],entity)
            #top_intent = 'numerical_reasoning'
        elif((score[0][2]>score[0][0])&(score[0][2]>score[0][3])&(score[0][2]>score[0][3])&(score[0][2]>score[0][4])&(score[0][2]>score[0][5])):
            entity.append('"status" : "Spatial_Visualization"')
            entity =  main_entity(query,capability_entity[1],entity)
            #top_intent = 'spatial Visualization '
        elif((score[0][3]>score[0][0])&(score[0][3]>score[0][2])&(score[0][3]>score[0][1])&(score[0][3]>score[0][4])&(score[0][3]>score[0][5])):
            entity.append('"status" : "Overall"')
            top_intent = 'overall'
        elif((score[0][4]>score[0][0])&(score[0][4]>score[0][2])&(score[0][4]>score[0][3])&(score[0][4]>score[0][1])&(score[0][4]>score[0][5])):
            entity.append('"status" : "Perceptual_Speed"')
            entity =  main_entity(query,capability_entity[1],entity)
            #top_intent = 'Perceptual Speed'
        else:
            entity.append('"status" : "Verbal_Reasoning"')
            entity =  main_entity(query,capability_entity[1],entity)
            #top_intent = 'verbal_reasoning'
    Score = str(Score)
    return ('{"TopIntent": '+top_intent+', "SubIntent": '+sub_intent +', "Percentage":'+Score,entity)

def intensity_search(query):
    intensity = ''
    if(re.search('fair',query)):
        intensity = 'fair'
    elif(re.search('talented',query)):
        intensity = 'talented'
    elif(re.search('proficient',query)):
        intensity = 'proficient'
    elif(re.search('exceptional',query)):
        intensity = 'exceptional'
    return intensity


def inquire(query,capability_entity,entity):
    inquiry = []
    if(re.search('percentile|score',query)):
        inquiry.append('percentile')
    if(re.search('strength| verbiage',query)):
        inquiry.append('strength')
    if(inquiry == []):
        m = load_model('D:\\bot\\botapi\\botapi\\models\\Capability_Assessment\\capability_assessment_sub_intent.h5py')
        score = model_score_LSTM_tokenize_stop(query,capability_entity,m)
        if(score[0][0]>score[0][1]):
            inquiry.append('percentile')
        else:
            inquiry.append('strength')
    return inquiry

def main_entity(query,capability_entity,entity):
    subintent = inquire(query,capability_entity,entity)
    intensity = intensity_search(query)
    print(subintent)
    if 'strength' in subintent:
        if(intensity == ''):
            entity.append('"Intensity" : "1"')
        else:
            entity.append('"Intensity" : "'+intensity+'"')
    else:
        entity.append('"Intensity" : "0"')
        percentile = str(percentile_extract(query))
        if(percentile == []):
            entity.append('"Percentile" : "1"')
        else:
            entity.append('"Percentile" : "'+percentile+'"')
    return entity







