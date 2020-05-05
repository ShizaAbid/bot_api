from keras.models import load_model
from os import path
import sys
from help_desk_func import token_stems
#from help_desk_func import model_score_LSTM_stop
from help_desk_func import model_score_LSTM_tokenize
import joblib

#sys.path.append('D:\\bot\\botapi\\botapi\\models\\Helpdesk\\')

def helpdesk_func(query,tok,tok_satisfication):
    top_intent = ''
    query = query.lower()
    m = load_model("models\\Helpdesk\\Helpdesk_main.h5py")
    score = model_score_LSTM_(query,tok,m)` `
    if((score[0][0]>score[0][1])&(score[0][0]>score[0][2])&(score[0][0]>score[0][3])):
        top_intent = 'Pending_Tickets'
    elif((score[0][1]>score[0][0])&(score[0][1]>score[0][2])&(score[0][1]>score[0][3])):
        top_intent = 'Satisfication_level'
        m = joblib.load("models\\Helpdesk\\satisfication_level_model.h5")
        return(model_score_LSTM(query,tok_satisfication,m))
    elif((score[0][2]>score[0][0])&(score[0][2]>score[0][1])&(score[0][2]>score[0][3])):
        top_intent = 'Summary_ticket_created'
    else:
        top_intent = 'Ticket_inquiry'
    return top_intent



