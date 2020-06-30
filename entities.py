#IMPORTING JUST IMPORTANT LIBRARIES
from keras.models import load_model
import nltk
import re
import spacy
from datetime import datetime
from datetime import timedelta
import datefinder
from dateparser.search import search_dates
from functions import model_score_LSTM
nlp = spacy.load('en_core_web_sm')

def entity_extract(name,tok_quantity,emp_detail):
    quantity = quantity_module(name,tok_quantity)
    entity = []
    entity.append('"Quantity":"'+ quantity + '"')
    leave_type = leavetype(name)
    entity.append(leave_type)
    name_entity = name_entity_extract(name,emp_detail)
    entity.append(name_entity)
    if(quantity == 'P'):
        if(name_entity == ''):
            entity.append('"Self":' + '"Y"')
    date = date_func(name)
    entity.append(date)
    while("" in entity):
        entity.remove("")

    return entity




                                                            ###### TO CHECK QUANTITY ######

def quantity_module(name,tok_quantity):
    sub_intent = ''
    m = load_model('D:\\bot\\botapi\\botapi\\models\\Quantity\\Quantity_Model.h5')
    score = model_score_LSTM(name,tok_quantity,m)
    if(score[0][0]>score[0][1]):
        sub_intent = sub_intent + "A"
    else:
        sub_intent = sub_intent + "P"
    return sub_intent
                                                            ###### FOR LEAVE TYPE ######


def leavetype(text):
    tokens=[word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens=[]
    for token in range(len(tokens)):
        if (re.search('earned|privilege|casual|sick|medical|half-pay|casual|maternity|quarantine|study|sabbatical|halfday|annual|hajj|official|special|disability|leave without pay|lwop', tokens[token])):
            filtered_tokens.append(str(tokens[token]+' leave').title()) #filtered_tokens #+ str(tokens[token]+' leave').title()+"*xx*"
        leavetypes=""
    if(len(filtered_tokens) != 0):
        if(len(filtered_tokens)>1):
            for leaves in range(len(filtered_tokens)-1):
                leavetypes = leavetypes + filtered_tokens[leaves] + "*xx*"
        leavetypes = leavetypes + filtered_tokens[(len(filtered_tokens))-1]
        return ('"Leave_Type": "'+leavetypes+'"')
    else:
        return ""




                                                            ###### FOR NAME ENTITY ######


def name_entity_extract(name,emp_detail):
    name_extract = ''
    name = name.title()
    nlp = spacy.load('en_core_web_sm')
    name_spacy = nlp(name)
    for num,sen in enumerate(name_spacy.sents):
        for ent in sen.ents:
            if ent.label_ == 'PERSON':
                print(ent.text)
                name_func = check_name(ent.text,emp_detail)
                name_extract = name_func
    if(name_extract == ''):
        name_func = check_name(name,emp_detail)
        name_extract = name_func
    if(name_extract != ''):
        return('"EmployeeDetail":'+'"'+str(name_extract)+'"')
    else:
        return ''






def check_name(text,emp_detail):
    text = text.lower()
    text = ' '+text+' '
    #name_database = pd.read_excel("SoftronicEmployee.xlsx")
    emp_code = emp_detail[1]
    emp_id = emp_detail[2]
    emp_name = emp_detail[0]
    #print(emp_name)
    correct_name = ''
    variations = ''
    #print(len(name_database))
    for a in range(len(emp_name)):
        name = str(emp_name[a])
        msname = "ms."+name
        mrname = "mr."+name
        name = name.lower()
        name_token = [word.lower() for sent in nltk.sent_tokenize(name) for word in nltk.word_tokenize(sent)]
        if(re.search(name,text)):
            correct_name = str(emp_name[a])+"@"+str(emp_code[a])+"@"+str(emp_id[a])
        elif(re.search(msname,text)):
            correct_name = str(emp_name[a])+"@"+str(emp_code[a])+"@"+str(emp_id[a])
        elif(re.search(mrname,text)):
            correct_name = str(emp_name[a])+"@"+str(emp_code[a])+"@"+str(emp_id[a])
        else:
            for token in range(len(name_token)):
                msname = "ms."+name_token[token]
                mrname = "mr."+name_token[token]
                if(re.search(' '+(str(name_token[token]))+' ',text)):
                    if(variations == ''):
                        variations = str(emp_id[a])+"@"+str(emp_code[a])+"@"+str(emp_name[a])
                    else:
                        variations = variations + ','+str(emp_id[a])+"@"+str(emp_code[a])+"@"+str(emp_name[a])
                elif(re.search(' '+msname+' ',text)):
                    if(variations == ''):
                        variations = str(emp_id[a])+"@"+str(emp_code[a])+"@"+str(emp_name[a])
                    else:
                        variations = variations + ','+str(emp_id[a])+"@"+str(emp_code[a])+"@"+str(emp_name[a])
                elif(re.search(' '+mrname+' ',text)):
                    if(variations == ''):
                        variations = str(emp_id[a])+"@"+str(emp_code[a])+"@"+str(emp_name[a])
                    else:
                        variations = variations + ','+str(emp_id[a])+"@"+str(emp_code[a])+"@"+str(emp_name[a])
    if(correct_name != ''):
        return(correct_name)
    else:
        return(variations)




                                                            ###### FOR DATE ######





def date_stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    tokens=[word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_sentence = []
    for w in range(len(tokens)):
        if tokens[w] not in stopwords:
            if(re.search('\d+%',tokens[w])):
                pass
            elif(re.search('\d+',tokens[w])):
                if(w+1 < len(tokens)):
                    if(re.search('%|percent',tokens[w+1])):
                        w = w + 1
                    else: filtered_sentence.append(tokens[w])
                else: filtered_sentence.append(tokens[w])
            else: filtered_sentence.append(tokens[w])
    filtered_sentence = " ".join(map(str,filtered_sentence))
    return filtered_sentence


def date_format(text):
    tokens=[word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    text = " ".join(map(str,tokens))
    corr = []
    text = nlp(text)
    text = [token.text for token in text]
    for token in range(len(text)):
        if (re.search('jan?|feb?|march|apr?|may|jun?|jul?|aug?|sep?|oct?|nov?|dec?',text[token])):
            if(re.search('jan |feb |mar |apr |jun |jul |aug |sep |oct |nov |dec ',text[token])):
                text[token]=datetime.strptime(text[token],'%b').strftime('%B')
            if(text[token-1]!='of'):
                if(re.search('[0-9]?',text[token-2])):
                    corr.append("of")
            corr.append(text[token].capitalize())
        else:
            corr.append(text[token])
    string = " ".join(map(str,corr))
    return string



def date_func(name):
    dates = ""
    string = date_format(name)
    stop_word = date_stopwords(name)
    date = search_dates(stop_word)
    print('Date time')
    print(date)
    if(date == None):
        date = ""
        return date
    for match in range(len(date)):
        if((match+1) == len(date)):
            dates = dates + str(date[match])
        else:
            dates = dates + str(date[match])+"*xxx*"
    dates = dates + "++"
    text = nlp(string)
    for num,sen in enumerate(text.sents):
        for ent in sen.ents:
            is_present = False
            is_date = search_dates(ent.text)
            if ent.label_ == 'DATE':
                dates = dates +(str(ent.text))+ "*xx*"
                st = ent.text
                for tok in st:
                    if(re.search('or|and|&',st)):
                        is_present = True
                if(is_present == True):
                    dates = dates + "*uxm*"
                elif(len(is_date)>1):
                    dates = dates + "*uxr*"
    dates = dates + "XXXXX"
    date = []
    matches = (datefinder.find_dates(string))
    for match in matches:
        date.append(match.strftime('%d-%m-%Y'))
    string=nlp(string)
    sentence = [token.text for token in string]
    for token in sentence:
        if (re.search('today|tomorrow|yesterday', token)):
            if(token == 'today'):
                token = datetime.today().strftime('%d-%m-%Y')
            elif(token == 'yesterday'):
                token = (datetime.now() - timedelta(days=1)).strftime('%d-%m-%Y')
            elif(token == 'tomorrow'):
                token = (datetime.now() + timedelta(days=1)).strftime('%d-%m-%Y')
            date.append(token)
    date.sort(key = lambda date: datetime.strptime(date, '%d-%m-%Y'))
    for d in date:
        dates = dates + d + "*xx*"
    if(dates != ""):
        return ('"datetime": "'+dates+'"')
    else:
        return ""


