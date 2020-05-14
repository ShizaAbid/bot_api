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
    Score = ''
    m = load_model('D:\\bot\\botapi\\botapi\\models\\Behavioral_Assesment\\Behavioral_model.h5py')
    score = model_score_LSTM_behavior(query,behavioral_entity[0],m)
    if((score[0][0]>score[0][1])&(score[0][0]>score[0][2])&(score[0][0]>score[0][3])):
        #top_intent = 'Stress'
        entity.append('"Status": "Stress"')
        Score = round(score[0][0]*100,2)
    elif((score[0][1]>score[0][0])&(score[0][1]>score[0][2])&(score[0][1]>score[0][3])):
        #top_intent = 'Concious_Persona'
        entity.append('"Status": "Concious_Persona"')
        if(persona_model(query,behavioral_entity[1]) == 'Graphs'):
            entity.append('"Graph : 1"')
        else:
            entity.append('"Graph : 0"')
        #entity.append(persona_model(query,behavioral_entity[1]))
        Score = round(score[0][1]*100,2)
    elif((score[0][2]>score[0][0])&(score[0][2]>score[0][3])&(score[0][2]>score[0][3])):
        #top_intent = 'Core_Persona'
        entity.append('"Status": "Core_Persona"')
        if(persona_model(query,behavioral_entity[1]) == 'Graphs'):
            entity.append('"Graph : "1"')
        else:
            entity.append('"Graph" : "0"')
        Score = round(score[0][2]*100,2)
    else:
        #top_intent = 'Flex_Graph'
        intent = Flex_persona(query,behavioral_entity[2])
        if(intent == 'Inquiry'):
            entity.append('"Status": "Inquiry"')
        else:
            entity.append('"Status": "Flex"')
            if(entity == 'Graphs'):
                entity.append('"Graph : "1"')
            else:
                entity.append('"Graph" : "0"')
                entity.append(intent)

        Score = round(score[0][3]*100,2)
    entity.append(energy_intensity(query))
    entity.append(energy_type(query))
    Score = str(Score)
    return ('{"TopIntent": "'+top_intent+'", "Percentage":'+Score,entity)

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
    m = load_model('D:\\Python_API\\bot_API_2\\bot_API_2\\models\\Behavioral_Assesment\\Flex_model.h5')
    score = model_score_LSTM_tokenize(query,flex_tok,m)
    intent =''
    if((score[0][0]>score[0][1])&(score[0][0]>score[0][2])&(score[0][0]>score[0][3])&(score[0][0]>score[0][4])):
        intent = '"shift" : "Change_Behavior"'
    elif((score[0][1]>score[0][0])&(score[0][1]>score[0][2])&(score[0][1]>score[0][3])&(score[0][1]>score[0][4])):
        if(re.search('graph',query)):
            intent = 'Graphs'
        else:
            intent = 'Inquiry'
    elif((score[0][2]>score[0][0])&(score[0][2]>score[0][3])&(score[0][2]>score[0][3])&(score[0][2]>score[0][4])):
        intent = "Inquiry"
    elif((score[0][3]>score[0][0])&(score[0][3]>score[0][2])&(score[0][3]>score[0][1])&(score[0][3]>score[0][4])):
        if(re.search('shift below',query)):
            intent = '"shift" : "Below"'
        else:
            intent = '"shift" : "Above"'
    else:
        if(re.search('shift above',query)):
            intent = '"shift" : "Above"'
        else:
            intent = '"shift" : "Below"'
    return intent
def energy_intensity(query):
    if(re.search('above mid[line| line]| high ',query)):
        return('"Intensity": "H"')
    elif(re.search('below mid[line| line]| low ',query)):
        return('"Intensity": "L"')
    else:
        return ('"Intensity": "0"')

def energy_type(query):
    energy = '"Energy": "'
    if(re.search(' red ',query)):
        energy = energy + 'R = 1**'
    else:
        energy = energy + 'R = 0**'
    if(re.search(' blue ',query)):
         energy = energy + 'B = 1**'
    else:
        energy = energy + 'B = 0**'
    if(re.search(' yellow ',query)):
        energy = energy + 'Y = 1**'
    else:
        energy = energy + 'Y = 0**'
    if(re.search(' green ',query)):
        energy = energy + 'G = 1'
    else:
        energy = energy + 'G = 0'
    energy = energy + '"'
    return energy
