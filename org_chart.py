from keras.models import load_model
import pandas as pd
from functions import token_stems
from functions import tok_behavior_flex
from functions import model_score_LSTM_tokenize


data = pd.read_excel("D:\\virtual\\Org chart.xlsx")
text = data['Data']
x = []
for a in text:
    x.append(token_stems(a))
tok = tok_behavior_flex(x)
m = load_model("C:\\Users\\shiza.abid\\Org_chart.h5py")

#FOR QUERY
query = "send me my organization chart"
score = model_score_LSTM_tokenize(query,tok,m)
if(score[0][0] > score [0][1]):
    print("Org chart")
else:
    print("JD")
print(score)

