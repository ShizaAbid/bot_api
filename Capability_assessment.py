from keras.models import load_model
from os import path
import sys
import numpy as np
from nltk.tokenize import word_tokenize
#from help_desk_func import token_stems
from functions import model_score_LSTM_tokenize
from leave import leave_func
from functions import model_score_LSTM_behavior
from functions import percentile_extract
#from help_desk_func import model_score_LSTM
from functions import model_score_LSTM_tokenize_stop
import joblib
from functions import percentile_change
import re

def capability_assessment_model(query,capability_entity,entity):
    top_intent = '"Assessment"'
    sub_intent = '"Capability_Assessment"'
    Score = 100
    report_check = 0
    graph_check = 0
    if(re.search('verbal interpretation',query)):
        entity.append('"Status" : "Verbal_Interpretation"')
        entity =  main_entity(query,capability_entity[1],entity)
    elif(re.search('numerical reasoning',query)):
        entity.append('"Status" : "Numerical_Reasoning"')
        entity =  main_entity(query,capability_entity[1],entity)
    elif(re.search('spatial visualization',query)):
        entity.append('"Status" : "Spatial_Visualization"')
        entity =  main_entity(query,capability_entity[1],entity)
    elif(re.search('perceptual speed',query)):
        entity.append('"Status" : "Perceptual_Speed"')
        entity =  main_entity(query,capability_entity[1],entity)
    elif(re.search('verbal reasoning',query)):
        entity.append('"Status" : "Verbal_Reasoning"')
        entity =  main_entity(query,capability_entity[1],entity)
    elif(re.search('capability (assessment|profile)',query)):
        intent = overall(query,capability_entity[2])
        if(intent == 'Inquiry'):
            entity.append('"Status": "Inquiry"')
        else:
            entity.append('"Status" : "Capability_Assessment"')
            if(intent == 'Report'):
                report_check = 1
                entity.append('"Report":"1"')
                entity.append('"Intensity" : "0"')
                entity.append('"Percentile" : 0"')
            elif(intent == 'Score'):
                entity =  main_entity(query,capability_entity[1],entity)
            elif(intent == 'Graph'):
                graph_check = 1
                entity.append('"Graph":"1"')
                entity.append('"Intensity" : "0"')
                entity.append('"Percentile" : 0"')
    else:
        m = load_model('D:\\bot\\botapi\\botapi\\models\\Capability_Assessment\\capability_assessment.h5py')
        score = model_score_LSTM_tokenize(query,capability_entity[0],m)
        if((score[0][0]>score[0][1])&(score[0][0]>score[0][2])&(score[0][0]>score[0][3])&(score[0][0]>score[0][4])&(score[0][0]>score[0][5])):
            entity.append('"Status" : "Verbal_Interpretation"')
            entity =  main_entity(query,capability_entity[1],entity)
        elif((score[0][1]>score[0][0])&(score[0][1]>score[0][2])&(score[0][1]>score[0][3])&(score[0][1]>score[0][4])&(score[0][1]>score[0][5])):
            entity.append('"Status" : "Numerical_Reasoning"')
            entity =  main_entity(query,capability_entity[1],entity)
            #top_intent = 'numerical_reasoning'
        elif((score[0][2]>score[0][0])&(score[0][2]>score[0][3])&(score[0][2]>score[0][3])&(score[0][2]>score[0][4])&(score[0][2]>score[0][5])):
            entity.append('"Status" : "Spatial_Visualization"')
            entity =  main_entity(query,capability_entity[1],entity)
            #top_intent = 'spatial Visualization '
        elif((score[0][3]>score[0][0])&(score[0][3]>score[0][2])&(score[0][3]>score[0][1])&(score[0][3]>score[0][4])&(score[0][3]>score[0][5])):
            entity.append('"Status" : "Overall"')
            intent = overall(query,capability_entity[2])
            if(intent == 'Inquiry'):
                entity.append('"Status": "Inquiry"')
            else:
                entity.append('"Status" : "Capability_Assessment"')
                if(intent == 'Report'):
                    report_check = 1
                    entity.append('"Report":"1"')
                    entity.append('"Intensity" : "0"')
                    entity.append('"Percentile" : 0"')
                elif(intent == 'Score'):
                    entity =  main_entity(query,capability_entity[1],entity)
                elif(intent == 'Graph'):
                    graph_check = 1
                    entity.append('"Graph":"1"')
                    entity.append('"Intensity" : "0"')
                    entity.append('"Percentile" : 0"')
        elif((score[0][4]>score[0][0])&(score[0][4]>score[0][2])&(score[0][4]>score[0][3])&(score[0][4]>score[0][1])&(score[0][4]>score[0][5])):
            entity.append('"Status" : "Perceptual_Speed"')
            entity =  main_entity(query,capability_entity[1],entity)
            #top_intent = 'Perceptual Speed'
        else:
            entity.append('"Status" : "Verbal_Reasoning"')
            entity =  main_entity(query,capability_entity[1],entity)
            #top_intent = 'verbal_reasoning'
    Score = str(Score)
    if(report_check == 0):
        entity.append('"Report":"0"')
    if(graph_check == 0):
        entity.append('"Graph":"0"')
    return ('{"TopIntent": '+top_intent+', "SubIntent": '+sub_intent +', "Percentage":'+Score,entity)

def intensity_search(query):
    intensity = ''
    if(re.search('fair',query)):
        intensity = 'Fair'
    elif(re.search('talented',query)):
        intensity = 'Talented'
    elif(re.search('proficient',query)):
        intensity = 'Proficient'
    elif(re.search('exceptional',query)):
        intensity = 'Exceptional'
    return intensity


def inquire(query,capability_entity):
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
    score_check = 0
    intensity_check = 0
    percentile_check = 0
    subintent = inquire(query,capability_entity)
    intensity = intensity_search(query)
    if 'strength' in subintent:
        intensity_check = 1
        if(intensity == ''):
            entity.append('"Intensity" : "1"')
        else:
            entity.append('"Intensity" : "'+intensity+'"')
    if 'percentile' in subintent:
        score_check = 1
        percentile = percentile_extract(query)
        if not percentile:
            entity.append('"Percentile" : "1"')
        else:
            entity.append('"Percentile" : "'+percentile+'"')
            check = percentile_change(query)
            if(check != []):
                percentile_check = 1
                entity.append(check)
    if(score_check == 0):
        entity.append('"Percentile" : "0"')
    if(intensity_check == 0):
        entity.append('"Intensity" : "0"')
    if(percentile_check == 0):
        entity.append('"Percentile_Change":"0"')
    return entity


def overall(query,overall_genism):
    inq = []
    if(re.search('percentile|score|strength|verbiage',query)):
        return('Score')
    if(re.search('report ',query)):
        return('Report')
    if(re.search('graph',query)):
        return('Graph')
    if not inq:
        print("here")
        sims_score = overall_genism[1]
        sims_report = overall_genism[4]
        sims_inquiry = overall_genism[7]
        tf_idf_score = overall_genism[2]
        tf_idf_report = overall_genism[5]
        tf_idf_inquiry = overall_genism[8]
        dictionary_score = overall_genism[3]
        dictionary_report = overall_genism[6]
        dictionary_inquiry = overall_genism[9]
        query_doc = [w.lower() for w in word_tokenize(query)]
        query_doc_bow = dictionary_score.doc2bow(query_doc)
        query_doc_tf_idf = tf_idf_score[query_doc_bow]
        score = np.max(sims_score[query_doc_tf_idf])
        print("Score:",np.max(sims_score[query_doc_tf_idf]))
        query_doc_bow = dictionary_report.doc2bow(query_doc)
        query_doc_tf_idf = tf_idf_report[query_doc_bow]
        report = np.max(sims_report[query_doc_tf_idf])
        print("Report:",np.max(sims_report[query_doc_tf_idf]))
        query_doc_bow = dictionary_inquiry.doc2bow(query_doc)
        query_doc_tf_idf = tf_idf_inquiry[query_doc_bow]
        inquiry = np.max(sims_inquiry[query_doc_tf_idf])
        print("Inquiry:",np.max(sims_inquiry[query_doc_tf_idf]))
        if((score>inquiry)&(score>report)):
            return('Score')
        elif((report>score)&(report>inquiry)):
            return('Report')
        else:
            return('Inquiry')








