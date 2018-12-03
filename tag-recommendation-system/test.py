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

from keras.models import load_model

model2=load_model('model_decision')
test=pd.read_csv('../input/he-toxic-multilabel/d583b256-d-new_dataset/new_dataset/test.csv')['id']
threshold=[.22]
loop=int(194323/batch_size)
remained=194323%batch_size
i=0

for thre in threshold:
    pre=[]
    while True:
        start=i*batch_size
        end=(i+1)*batch_size
        if i==loop:
            end=i*batch_size+remained
            batch_size=remained
            print(start,end)
        pred=model2.predict(x_test[start:end])
        ddx=np.argmax(pred,axis=1)
        pred[np.arange(batch_size),ddx]=1
        pred[pred>=thre]=1
        pred[pred<thre]=0
        tagss=mlb.inverse_transform(pred)
        for tg,idx in zip(tagss,test.values[start:end]):
            strn=''
            for wo in tg:
                strn=strn+wo+'|'
            pre.append({'id':idx,'tags':strn[:-1]})
        i+=1

        if i>loop:
            break


pd.DataFrame(pre).to_csv("submission.csv",index=False)