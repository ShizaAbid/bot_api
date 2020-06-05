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

def behavioral_assessment_model(query,behavioral_entity,entity):
    top_intent = '"Assessment"'
    sub_intent = '"Behavioral_Assessment"'
    intent  = ''
    Score = ''
    if(re.search('stress',query)):
        entity.append('"Status": "Stress"')
        Score = 100
        if(re.search(' dont | arent| aren\'t| don\'t| not',query)):
            entity.append('"Negative": "0"')
        else:
            entity.append('"Negative": "1"')
    elif(re.search('conscious persona| working enviroment',query)):
        entity.append('"Status": "Conscious_Persona"')
        Score = 100
        entity.append('"Negative": "1"')
        if(persona_model(query,behavioral_entity[1]) == 'Graphs'):
            entity.append('"Graph" : "1"')
        else:
            entity.append('"Graph" : "0"')
    elif(re.search('core persona| natural enviroment',query)):
        entity.append('"Status": "Core_Persona"')
        entity.append('"Negative": "1"')
        Score = 100
        if(persona_model(query,behavioral_entity[1]) == 'Graphs'):
            entity.append('"Graph" : "1"')
        else:
            entity.append('"Graph" : "0"')
    elif(re.search(' flex|shift above|shift below|shifted above|shifted below|((change|changed|modify|modifies|modified)( my| her| his| there| their)* behavior)',query)):
        intent = Flex_persona(query,behavioral_entity[2])
        entity.append('"Negative": "1"')
        Score = 100
        if(intent == 'Inquiry'):
            entity.append('"Status": "Inquiry"')
        else:
            entity.append('"Status": "Flex"')
            if(intent == 'Graphs'):
                entity.append('"Graph" : "1"')
            else:
                entity.append('"Graph" : "0"')
                print(intent)
                entity.append(intent)
    else:
        m = load_model('D:\\bot\\botapi\\botapi\\models\\Behavioral_Assesment\\Behavioral_model.h5py')
        score = model_score_LSTM_behavior(query,behavioral_entity[0],m)
        if((score[0][0]>score[0][1])&(score[0][0]>score[0][2])&(score[0][0]>score[0][3])):
            #top_intent = 'Stress'
            entity.append('"Status": "Stress"')
            if(re.search(' dont | arent| aren\'t| don\'t| not',query)):
                entity.append('"Negative": "0"')
            else:
                entity.append('"Negative": "1"')
            Score = round(score[0][0]*100,2)
        elif((score[0][1]>score[0][0])&(score[0][1]>score[0][2])&(score[0][1]>score[0][3])):
            #top_intent = 'Concious_Persona'
            entity.append('"Status": "Conscious_Persona"')
            entity.append('"Negative": "1"')
            intent = persona_model(query,behavioral_entity[1])
            if( intent == 'Graphs'):
                entity.append('"Graph" : "1"')
            else:
                entity.append('"Graph" : "0"')
            #entity.append(persona_model(query,behavioral_entity[1]))
            Score = round(score[0][1]*100,2)
        elif((score[0][2]>score[0][0])&(score[0][2]>score[0][3])&(score[0][2]>score[0][3])):
            #top_intent = 'Core_Persona'
            entity.append('"Status": "Core_Persona"')
            entity.append('"Negative": "1"')
            if(persona_model(query,behavioral_entity[1]) == 'Graphs'):
                entity.append('"Graph" : "1"')
            else:
                entity.append('"Graph" : "0"')
            Score = round(score[0][2]*100,2)
        else:
            #top_intent = 'Flex_Graph'
            intent = Flex_persona(query,behavioral_entity[2])
            entity.append('"Negative": "1"')
            if(intent == 'Inquiry'):
                if(re.search('behavioral (assessment )*report',query)):
                    entity.append('"Status" : "Report"')
                else:
                    entity.append('"Status": "Inquiry"')
            else:
                entity.append('"Status": "Flex"')
                if(intent == 'Graphs'):
                    entity.append('"Graph" : "1"')
                else:
                    entity.append('"Graph" : "0"')
                    entity.append(intent)
            Score = round(score[0][3]*100,2)

    entity.append(energy_intensity(query))
    entity = energy_type(query,entity)
    Score = str(Score)
    return ('{"TopIntent": '+top_intent+', "SubIntent": '+sub_intent +', "Percentage":'+Score,entity)

def persona_model(query,persona_tok):
    intent = ''
    m = load_model('D:\\bot\\botapi\\botapi\\models\\Behavioral_Assesment\\Assessment_concious_core_persona_Model.h5')
    score = model_score_LSTM_tokenize(query,persona_tok,m)
    if((score[0][0]>score[0][1])&(score[0][0]>score[0][2])&(score[0][0]>score[0][3])):
        intent = 'Graphs'
    elif((score[0][1]>score[0][0])&(score[0][1]>score[0][2])&(score[0][1]>score[0][3])):
        intent = 'Inquire_Energy'
    elif((score[0][2]>score[0][0])&(score[0][2]>score[0][3])&(score[0][2]>score[0][3])):
        intent = 'Inquire_Intesity'
    else:
        intent = 'Intensity'
    return intent

def Flex_persona(query,flex_tok):
    intent =''
    if(re.search('shift',query)):
        if(re.search(' above| up',query)):
            intent = '"Shift" : "Above"'
        if(re.search(' below| down',query)):
            intent = '"Shift" : "Below"'
    else:
        m = load_model('D:\\bot\\botapi\\botapi\\models\\Behavioral_Assesment\\Flex_model.h5')
        score = model_score_LSTM_tokenize(query,flex_tok,m)
        if((score[0][0]>score[0][1])&(score[0][0]>score[0][2])&(score[0][0]>score[0][3])&(score[0][0]>score[0][4])):
            intent = '"Shift" : "Change_Behavior"'
        elif((score[0][1]>score[0][0])&(score[0][1]>score[0][2])&(score[0][1]>score[0][3])&(score[0][1]>score[0][4])):
            if(re.search(' graph| flex',query)):
                intent = 'Graphs'
            else:
                intent = 'Inquiry'
        elif((score[0][2]>score[0][0])&(score[0][2]>score[0][3])&(score[0][2]>score[0][3])&(score[0][2]>score[0][4])):
            intent = "Inquiry"
        elif((score[0][3]>score[0][0])&(score[0][3]>score[0][2])&(score[0][3]>score[0][1])&(score[0][3]>score[0][4])):
            if(re.search('shift below',query)):
                intent = '"Shift" : "Below"'
            else:
                intent = '"Shift" : "Above"'
        else:
            if(re.search('shift above',query)):
                intent = '"Shift" : "Above"'
            else:
                intent = '"Shift" : "Below"'
    return intent

def energy_intensity(query):
    if(re.search('(above mid(line| line))| high',query)):
        return('"Intensity": "H"')
    elif(re.search('(below mid(line| line))| low',query)):
        return('"Intensity": "L"')
    else:
        return ('"Intensity": "1"')

def energy_type(query,entity):
    energy = '"Energy": "'
    if(re.search(' red ',query)):
        energy = '"R" : "1"'
    else:
        energy = '"R" : "0"'
    entity.append(energy)
    if(re.search(' blue ',query)):
        energy = '"B" : "1"'
    else:
        energy = '"B" : "0"'
    entity.append(energy)
    if(re.search(' yellow ',query)):
        energy = '"Y" : "1"'
    else:
        energy = '"Y" : "0"'
    entity.append(energy)

    if(re.search(' green ',query)):
        energy = '"G" : "1"'
    else:
        energy = '"G" : "0"'
    entity.append(energy)
    return entity
