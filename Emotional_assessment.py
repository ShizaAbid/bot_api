import re
from keras.models import load_model
from os import path
import sys
#from help_desk_func import token_stems
from functions import model_score_LSTM_tokenize
from leave import leave_func
from functions import model_score_LSTM_behavior
from functions import model_score_LSTM
from nltk.tokenize import word_tokenize
import numpy as np
#from help_desk_func import model_score_LSTM
import joblib
from functions import percentile_extract
from functions import percentile_change
import re


def emotional_assessment_model(query,EA_tok,entity):
    report_check = 0
    graph_check = 0
    score_check = 0
    top_intent = '"Assessment"'
    sub_intent = '"Emotional_Assessment"'
    Score = '100'
    if(re.search('emotional mindset|happiness|optimism|(self( |-)esteem)|(self( |-)motivation)',query)):
        entity = emotional_mindset_func(query,EA_tok[1],entity)
    elif(re.search('(understanding self (&|and) other)|understanding self|empathy',query)):
        entity = understanding_self_and_others_func(query,EA_tok[2],entity)
    elif(re.search('mastering self with other|assertiveness|influence|managing relationship|decision confidence|emotional expression',query)):
        entity = mastering_self_with_others(query,EA_tok[3],entity)
    elif(re.search('mastering self|stress managment|impulse control|emotional self control|bias managment|change managment',query)):
        entity = mastering_self_func(query,EA_tok[4],entity)
    elif(re.search('(emotional (assessment|intelligence|profile))|eq',query)):
        entity.append('"Status": "Emotional_Assessment"')
    else:
        m= load_model('D:\\bot\\botapi\\botapi\\models\\Emotional_Assessment\\Emotional_main.h5')
        score = model_score_LSTM(query,EA_tok[0],m)
        if((score[0][0]>score[0][1])&(score[0][0]>score[0][2])&(score[0][0]>score[0][3])&(score[0][0]>score[0][4])):
            entity = mastering_self_func(query,EA_tok[4],entity)
        elif((score[0][1]>score[0][0])&(score[0][1]>score[0][2])&(score[0][1]>score[0][3])&(score[0][1]>score[0][4])):
            entity = mastering_self_with_others(query,EA_tok[3],entity)
        elif((score[0][2]>score[0][0])&(score[0][2]>score[0][1])&(score[0][2]>score[0][3])&(score[0][2]>score[0][4])):
            entity = understanding_self_and_others_func(query,EA_tok[2],entity)
        elif((score[0][3]>score[0][0])&(score[0][3]>score[0][1])&(score[0][3]>score[0][2])&(score[0][3]>score[0][4])):
            entity = emotional_mindset_func(query,EA_tok[1],entity)
        else:
            entity.append('"Status": "Emotional_Assessment"')
    intent = overall(query,EA_tok[5])
    print(intent)
    if 'Report' in intent:
        report_check = 1
        entity.append('"Report":"1"')
    if 'Score' in intent:
        score_check = 1
        entity = main_entity(query,entity)
    if 'Graph' in intent:
        graph_check = 1
        entity.append('"Graph":"1"')
    if(score_check == 0):
        entity.append('"Percentile" : "0"')
        entity.append('"Intensity" : "0"')
    if(graph_check == 0):
        entity.append('"Graph":"0"')
    if(report_check == 0):
        entity.append('"Report":"0"')



    return ('{"TopIntent": '+top_intent+', "SubIntent": '+sub_intent +', "Percentage":'+Score,entity)

def emotional_mindset_func(query,tok,entities):
    if(re.search('happiness',query)):
        entities.append('"Status": "Happiness"')
    elif(re.search('optimism',query)):
        entities.append('"Status": "Optimism"')
    elif(re.search('(self(-| )esteem)',query)):
        entities.append('"Status": "Self_Esteem"')
    elif(re.search('(self(-| )motivation)',query)):
        entities.append('"Status": "Self_Motivation"')
    elif(re.search('(emotional (assessment|intelligence|profile))|eq',query)):
        entities.append('"Status": "Emotional_Assessment"')
    else:
        m = load_model('D:\\bot\\botapi\\botapi\\models\\Emotional_Assessment\\Emotional_Mindset_Model.h5')
        score = model_score_LSTM_tokenize(query,tok,m)
        if((score[0][0]>score[0][1])&(score[0][0]>score[0][2])&(score[0][0]>score[0][3])&(score[0][0]>score[0][4])):
            entities.append('"Status": "Happiness"')
        elif((score[0][1]>score[0][0])&(score[0][1]>score[0][2])&(score[0][1]>score[0][3])&(score[0][1]>score[0][4])):
            if(re.search('optimisim|optimistic',query)):
                entities.append('"Status": "Optimism"')
            else:
                entities.append('"Status": "Emotional_Mindset"')
        elif((score[0][2]>score[0][0])&(score[0][2]>score[0][1])&(score[0][2]>score[0][3])&(score[0][2]>score[0][4])):
            entities.append('"Status": "Emotional_Mindset"')
        elif((score[0][3]>score[0][0])&(score[0][3]>score[0][1])&(score[0][3]>score[0][2])&(score[0][3]>score[0][4])):
            entities.append('"Status": "Self_Esteem"')
        else:
            entities.append('"Status": "Self_Motivation"')
    return entities



def understanding_self_and_others_func(query,tok,entities):
    if(re.search('empathy',query)):
        entities.append('"Status": "Empathy"')
    elif(re.search('(understanding self (and|&) other)',query)):
        entities.append('"Status": "Understanding_Self_And_Others"')
    elif(re.search('understanding self',query)):
        entities.append('"Status": "Understanding_Self"')
    else:
        m = load_model('D:\\bot\\botapi\\botapi\\models\\Emotional_Assessment\\Understanding_Self_And_Others_Model.h5')
        score = model_score_LSTM_tokenize(query,tok,m)
        if((score[0][0]>score[0][1])&(score[0][0]>score[0][2])):
            entities.append('"Status": "Empathy"')
        elif((score[0][1]>score[0][0])&(score[0][1]>score[0][2])):
            entities.append('"Status": "Understanding_Self_And_Others"')
        else:
            entities.append('"Status": "Understanding_Self"')
    return entities

def mastering_self_with_others(query,tok,entities):
    if(re.search('assertiveness',query)):
        entities.append('"Status": "Assertiveness"')
    elif(re.search('influence',query)):
        entities.append('"Status": "Influence"')
    elif(re.search('managing relationship',query)):
        entities.append('"Status": "Managing_Relationships"')
    elif(re.search('decision confidence',query)):
        entities.append('"Status": "Decision_Confidence"')
    elif(re.search('emotional expression',query)):
        entities.append('"Status": "Emotional_Expression"')
    else:
        m= load_model('D:\\bot\\botapi\\botapi\\models\\Emotional_Assessment\\Mastering_self_with_others.h5')
        score = model_score_LSTM_tokenize(query,tok,m)
        if((score[0][0]>score[0][1])&(score[0][0]>score[0][2])&(score[0][0]>score[0][3])&(score[0][0]>score[0][4])&(score[0][0]>score[0][5])):
            entities.append('"Status": "Assertiveness"')
        elif((score[0][1]>score[0][0])&(score[0][1]>score[0][2])&(score[0][1]>score[0][3])&(score[0][1]>score[0][4])&(score[0][1]>score[0][5])):
            entities.append('"Status": "Decision_Confidence"')
        elif((score[0][2]>score[0][0])&(score[0][2]>score[0][1])&(score[0][2]>score[0][3])&(score[0][2]>score[0][4])&(score[0][2]>score[0][5])):
            entities.append('"Status": "Emotional_Expression"')
        elif((score[0][3]>score[0][0])&(score[0][3]>score[0][1])&(score[0][3]>score[0][2])&(score[0][3]>score[0][4])&(score[0][3]>score[0][5])):
            entities.append('"Status": "Influence"')
        elif((score[0][4]>score[0][0])&(score[0][4]>score[0][1])&(score[0][4]>score[0][2])&(score[0][4]>score[0][3])&(score[0][4]>score[0][5])):
            entities.append('"Status": "Mastering_Self_With_Others"')
        else:
            entities.append('"Status": "Managing_Relationships"')
    return entities

def mastering_self_func(query,tok,entities):
    if(re.search('stress managment',query)):
        entities.append('"Status": "Stress_Managment"')
    elif(re.search('impulse control',query)):
        entities.append('"Status": "Impulse_Control"')
    elif(re.search('emotional self control',query)):
        entities.append('"Status": "Emotional_Self_Control"')
    elif(re.search('bias managment',query)):
        entities.append('"Status": "Bias_Managment"')
    elif(re.search('change managment',query)):
        entities.append('"Status": "Change_Managment"')
    else:
        m= load_model('D:\\bot\\botapi\\botapi\\models\\Emotional_Assessment\\Assessment_emotional_mastering_self.h5')
        score = model_score_LSTM_tokenize(query,tok,m)
        if((score[0][0]>score[0][1])&(score[0][0]>score[0][2])&(score[0][0]>score[0][3])&(score[0][0]>score[0][4])&(score[0][0]>score[0][5])):
            entities.append('"Status": "Bias_Managment"')
        elif((score[0][1]>score[0][0])&(score[0][1]>score[0][2])&(score[0][1]>score[0][3])&(score[0][1]>score[0][4])&(score[0][1]>score[0][5])):
            entities.append('"Status": "Change_Managment"')
        elif((score[0][2]>score[0][0])&(score[0][2]>score[0][1])&(score[0][2]>score[0][3])&(score[0][2]>score[0][4])&(score[0][2]>score[0][5])):
            entities.append('"Status": "Emotional_Self_Control"')
        elif((score[0][3]>score[0][0])&(score[0][3]>score[0][1])&(score[0][3]>score[0][2])&(score[0][3]>score[0][4])&(score[0][3]>score[0][5])):
            entities.append('"Status": "Impulse_Control"')
        elif((score[0][4]>score[0][0])&(score[0][4]>score[0][1])&(score[0][4]>score[0][2])&(score[0][4]>score[0][3])&(score[0][4]>score[0][5])):
            entities.append('"Status": "Mastering_Self"')
        else:
            entities.append('"Status": "Stress_Managment"')
    return entities

def overall(query,overall_genism):
    inq = []
    if(re.search('percentile|score|strength|verbiage',query)):
        inq.append('Score')
    if(re.search('report ',query)):
        inq.append('Report')
    if(re.search('graph',query)):
        inq.append('Graph')
    if not inq:
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
    return(inq)


def intensity_search(query):
    intensity = ''
    if(re.search(' low',query)):
        intensity = 'Low'
    elif(re.search('average',query)):
        intensity = 'Average'
    elif(re.search('high',query)):
        intensity = 'High'
    return intensity


def inquire(query):
    inquiry = []
    if(re.search('percentile|score',query)):
        inquiry.append('percentile')
    if(re.search('strength| verbiage',query)):
        inquiry.append('strength')
    if(inquiry == []):
            inquiry.append('percentile')
            inquiry.append('strength')
    return inquiry

def main_entity(query,entity):
    score_check = 0
    intensity_check = 0
    percentile_check = 0
    subintent = inquire(query)
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


