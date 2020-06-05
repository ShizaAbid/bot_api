from keras.models import load_model
from os import path
import sys
#from help_desk_func import token_stems
from functions import model_score_LSTM_stop
from leave import leave_func
from Separation import separation_func
from behavioral_assessment import behavioral_assessment_model
from Capability_assessment import capability_assessment_model
from Emotional_assessment import emotional_assessment_model
#from help_desk_func import model_score_LSTM
import joblib
from functions import model_score_LSTM
import re

sys.path.append('D:\\bot\\botapi\\botapi\\models')

def mainmodel_func(query,tok,module_tok,entity):
    top_intent = ''
    query = query.lower()
    if(re.search('verbal interpretation|numerical reasoning|spatial visualization|perceptual speed|verbal reasoning|capability [assessment|profile]',query)):
        top_intent = 'capability_module'
        print(top_intent)
        top_intent = capability_assessment_model(query,module_tok[3],entity)
    elif(re.search('emotional mindset|happiness|optimism|self esteem|self motivation|understading self and others|understanding self & others|understanding self|empathy|mastering self with others|assertiveness|influence|managing relationships|decision confidence|emotional expression|mastering self|stress managment|impulse control|emotional self control|bias managment|change managment',query)):
        top_intent = 'emotional assessment'
        top_intent = emotional_assessment_model(query,module_tok[4],entity)
    elif(re.search('stress |conscious persona| working enviroment|core persona| natural enviroment|flex|shift above|shift below|shifted above|shifted below|[change|changed][my|her|his|there|their]* behavior|[change|changed][ |my|her|his|there|their]* behaviour|[modified|modify] there|behavioral profile',query)):
        top_intent = 'Behavioral_Module'
        top_intent = behavioral_assessment_model(query,module_tok[2],entity)
    else:
        m = load_model("D:\\bot\\botapi\\botapi\\models\\Main\\Main_Model.h5")
        score = model_score_LSTM(query,tok,m)
        print(score)
        if((score[0][0]>score[0][1])&(score[0][0]>score[0][2])&(score[0][0]>score[0][3])&(score[0][0]>score[0][4])):
            top_intent = 'Behavioral_Module'
            top_intent = behavioral_assessment_model(query,module_tok[2],entity)
        elif((score[0][1]>score[0][0])&(score[0][1]>score[0][2])&(score[0][1]>score[0][3])&(score[0][1]>score[0][4])):
            top_intent = 'capability_module'
            print(top_intent)
            top_intent = capability_assessment_model(query,module_tok[3],entity)
        elif((score[0][2]>score[0][0])&(score[0][2]>score[0][1])&(score[0][2]>score[0][3])&(score[0][2]>score[0][4])):
            top_intent = 'emotional assessment'
            top_intent = emotional_assessment_model(query,module_tok[4],entity)
        elif((score[0][3]>score[0][0])&(score[0][3]>score[0][1])&(score[0][3]>score[0][2])&(score[0][3]>score[0][4])):
            top_intent = 'Leave'
            top_intent = leave_func(query,module_tok[0],entity)
        else:
            top_intent = 'Separation'
            top_intent = separation_func(query,module_tok[1],entity)
    return top_intent