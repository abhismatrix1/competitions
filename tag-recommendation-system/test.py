# this file is used to get output from trained model in csv format for the submission
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Embedding, MaxPooling1D, Conv1D, SpatialDropout1D,Lambda, Activation
from keras.layers import add, Dropout, PReLU, BatchNormalization, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import optimizers
from keras import initializers, regularizers, constraints, callbacks
from keras import backend as K
import pandas as pd
import numpy as np
from keras.models import load_model, model_from_json
from preprocess import preprocess_data
from model import model_feature,model_decision
from model import difference
from util import precision, recall, fbeta_score,focal_loss,custom_loss
import gc
import pickle
from tqdm import tqdm

maxlen=200
batch_size=1024
#model2=load_model('model/model_decision',custom_objects={'tf':tf,'fbeta_score':fbeta_score,'precision':precision,'recall':recall,'custom_loss':custom_loss})
model2=load_model('model/model_decision',custom_objects={'tf':tf,'fbeta_score':fbeta_score,'precision':precision,'recall':recall,'focal_loss_fixed':focal_loss()})
print('model loaded. Now loading data...')
with open('tokenizer.pickle', 'rb') as handle:
	tokenizer = pickle.load(handle)


#model1=load_model('model/model_feature')
test=pd.read_csv('data/Dataset-DL4/test.csv')
test=test.drop(['id'],axis=1)
test['article']=test['article'].str.replace('</p>|<p>|\r|\n|<br>|</p>|<pre>|</pre>|<code>|</code>','')
test['combined']=test['title']+' '+test['article']
test.drop(['title','article'],axis=1,inplace=True)
X_test = test["combined"].fillna("fillna").values
X_test = tokenizer.texts_to_sequences(X_test)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
gc.collect()
del X_test,tokenizer
print("tokenizering done")
with open('multi_label_binarizer.pickle', 'rb') as handle:
	mlb = pickle.load(handle)
print("data loaded.Now starting evaluation...")
test=pd.read_csv('data/Dataset-DL4/test.csv')['id']
threshold=.22
BB=.25
loop=int(194323/batch_size)
remained=194323%batch_size
i=0
pre=[]
pbar = tqdm(total = loop)

while True:
    start=i*batch_size
    end=(i+1)*batch_size
    if i==loop:
        end=i*batch_size+remained
        batch_size=remained

    pred=model2.predict(x_test[start:end])

    ddx=np.argmax(pred,axis=1)
    pred[np.arange(batch_size),ddx]=1
    pred[pred>=threshold]=1
    pred[pred<threshold]=0
    
    tagss=mlb.inverse_transform(pred)
    for tg,idx in zip(tagss,test.values[start:end]):
        strn=''
        for wo in tg:
            strn=strn+wo+'|'
        pre.append({'id':idx,'tags':strn[:-1]})
    i+=1

    pbar.update(1)
    if i>loop:
        break


pd.DataFrame(pre).to_csv("submission.csv",index=False)

