from keras.models import load_model
from os import path
import sys
from functions import model_score_LSTM
from nltk.tokenize import word_tokenize
import numpy as np
import re

#sys.path.append('D:\\bot\\botapi\\botapi\\models\\Helpdesk\\')

def leave_func(query,leave_entities,entities):
    top_intent = ''
    query = query.lower()
    m = load_model("D:\\bot\\botapi\\botapi\\models\\Leave\\Leave_Model.h5")
    score = model_score_LSTM(query,leave_entities[0],m)
    if((score[0][0]>score[0][1])&(score[0][0]>score[0][2])&(score[0][0]>score[0][3])):
        Score = round(score[0][0]*100,2)
        top_intent = 'Leave_Request'
    elif((score[0][1]>score[0][0])&(score[0][1]>score[0][2])&(score[0][1]>score[0][3])):
        top_intent = 'Leave_Approval'
        Score = round(score[0][1]*100,2)
        entities.append(leave_approval(query,leave_entities[2]))
        #print(leave_approval(query,leave_entities[2]))
    elif((score[0][2]>score[0][0])&(score[0][2]>score[0][1])&(score[0][2]>score[0][3])):
        Score = round(score[0][2]*100,2)
        top_intent="Leave_Request"
        leave_type = emotional_leave(query,leave_entities[1])
        if leave_type not in entities:
            entities.append(leave_type)
    else:
        Score = round(score[0][3]*100,2)
        top_intent = 'Leave_Inquiry'
    Score = str(Score)
    return ('{"TopIntent": "'+top_intent+'", "Percentage":'+Score,entities)

def emotional_leave(name,leave_emotional_genism):
    query = name
    sims_s =leave_emotional_genism[1]
    sims_c = leave_emotional_genism[4]
    tf_idf_s = leave_emotional_genism[2]
    tf_idf_c = leave_emotional_genism [5]
    dictionary_s = leave_emotional_genism [3]
    dictionary_c = leave_emotional_genism [6]
    query_doc = [w.lower() for w in word_tokenize(query)]
    query_doc_bow = dictionary_s.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf_s[query_doc_bow]
    sick = np.max(sims_s[query_doc_tf_idf])
    print("sick:",np.max(sims_s[query_doc_tf_idf]))
    query_doc_bow = dictionary_c.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf_c[query_doc_bow]
    casual = np.max(sims_c[query_doc_tf_idf])
    print("Casual:",np.max(sims_c[query_doc_tf_idf]))
    if(casual>sick):
        return('"Leave_Type": "Casual Leave"')
    else:
        return('"Leave_Type": "Sick Leave"')


def leave_approval(name,leave_approval_genism):
    query = name
    sub_type = ''
    sims_approve = leave_approval_genism[1]
    tf_idf_approve = leave_approval_genism[2]
    dictionary_approve = leave_approval_genism[3]
    sims_reject = leave_approval_genism[4]
    tf_idf_reject = leave_approval_genism [5]
    dictionary_reject = leave_approval_genism[6]
    query_doc = [w.lower() for w in word_tokenize(query)]
    query_doc_bow = dictionary_approve.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf_approve[query_doc_bow]
    approve = np.max(sims_approve[query_doc_tf_idf])
    query_doc_bow = dictionary_reject.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf_reject[query_doc_bow]
    reject = np.max(sims_reject[query_doc_tf_idf])

    if(approve > reject):
        sub_type = sub_type + "approve"
    else:
        sub_type = sub_type + "disapprove"
    if(re.search('not approve|not accept|not acept|reject|disapprove|dis approve',name)):
        sub_type = "disapprove"
    elif(re.search('approve|accept|acept',name)):
        sub_type = "approve"
    sub_type = '"Status":' + '"' + sub_type + '"'
    return(str(sub_type))