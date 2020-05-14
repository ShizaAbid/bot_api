from keras.models import load_model
from os import path
import sys
#from help_desk_func import token_stems
from functions import model_score_LSTM_stop
from leave import leave_func
from Separation import separation_func
from behavioral_assessment import behavioral_assessment_model
from Capability_assessment import capability_assessment_model
#from help_desk_func import model_score_LSTM
import joblib
from functions import model_score_LSTM_tokenize

sys.path.append('D:\\bot\\botapi\\botapi\\models')

def mainmodel_func(query,tok,module_tok,entity):
    top_intent = ''
    query = query.lower()
    m = load_model("D:\\bot\\botapi\\botapi\\models\\Main\\Main_Model.h5")
    score = model_score_LSTM_tokenize(query,tok,m)
    if((score[0][0]>score[0][1])&(score[0][0]>score[0][2])&(score[0][0]>score[0][3])&(score[0][0]>score[0][4])):
        top_intent = 'Behavioral_Module'
        top_intent = behavioral_assessment_model(query,module_tok[2],entity)
    elif((score[0][1]>score[0][0])&(score[0][1]>score[0][2])&(score[0][1]>score[0][3])&(score[0][1]>score[0][4])):
        top_intent = 'capability_module'
        print(top_intent)
        top_intent = capability_assessment_model(query,module_tok[3],entity)
    elif((score[0][2]>score[0][0])&(score[0][2]>score[0][1])&(score[0][2]>score[0][3])&(score[0][2]>score[0][4])):
        top_intent = 'emotional assessment'
    elif((score[0][3]>score[0][0])&(score[0][3]>score[0][1])&(score[0][3]>score[0][2])&(score[0][3]>score[0][4])):
        top_intent = 'Leave'
        top_intent = leave_func(query,module_tok[0],entity)
    else:
        top_intent = 'Separation'
        top_intent = separation_func(query,module_tok[1],entity)
    return top_intent