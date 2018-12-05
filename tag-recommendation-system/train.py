
import os
import gc
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')


from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.cross_validation import KFold
from keras.models import Model
from keras.layers import Input, Dense, Embedding, MaxPooling1D, Conv1D, SpatialDropout1D
from keras.layers import add, Dropout, PReLU, BatchNormalization, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import optimizers
from keras import initializers, regularizers, constraints, callbacks
from keras import backend as K
from keras.utils import Sequence
from model import model_feature,model_decision
from preprocess import preprocess_data
from keras.models import load_model
from util import fbeta_score,precision,recall

BATCH_SIZE = 512

def train(embedding_matrix, x_train, y_train,x_test, mlb):
	model=model_feature(embedding_matrix)
	lr = callbacks.LearningRateScheduler(schedule)
	xtrain_gen, eval_gen, steps_per_epoch, validation_steps=create_generator(BATCH_SIZE)
	del x_train, y_train
	checkpoint=callbacks.ModelCheckpoint('model_tag',save_best_only=True)
	model.compile(loss='binary_crossentropy', 
	            optimizer=optimizers.Adam(lr=.001),
	            metrics=[fbeta_score])
	            
	epochs = 1
	model.fit_generator(xtrain_gen,
	                        steps_per_epoch=steps_per_epoch, epochs=epochs,
	                       validation_data=eval_gen,
	                        validation_steps=validation_steps,
	                         verbose=1,
	                        use_multiprocessing=False,
	                            callbacks=[lr,checkpoint]
	                       )
	model.load_weights('model_tag')
	model2=model_decision(model)
	model2.compile(loss='binary_crossentropy', 
	            optimizer=optimizers.Adam(lr=.001),
	            metrics=[fbeta_score,precision,recall])
	            

	epochs = 1
	checkpoint2=callbacks.ModelCheckpoint('model_decision',save_best_only=True)
	model2.fit_generator(xtrain_gen,
	                        steps_per_epoch=ytrain.shape[0] // batch_size, epochs=epochs,
	                       validation_data=eval_gen,
	                        validation_steps=yval.shape[0] // batch_size,
	                         verbose=1,
	                        use_multiprocessing=False,
	                            callbacks=[lr,checkpoint2]
	                       )

	model2.load_weights('model_decision')

	generate_submission_file(x_test)

	return


def create_generator(batch_size):
	Xtrain, Xval, ytrain, yval = train_test_split(x_train, y_train, train_size=0.98, random_state=233)
	xtrain_gen=dgen(x_train,y_train,batch_size)
	eval_gen=dgen(Xval,yval,batch_size)
	steps_per_epoch=ytrain.shape[0] // batch_size
	validation_steps=yval.shape[0] // batch_size
	return xtrain_gen, eval_gen, steps_per_epoch, validation_steps

def schedule(ind):
    indx=int(ind/8)
    a = [ 0.001, 0.0005, 0.0003, 0.0002,0.0001,0.0001,0.00005]
    return a[indx] 

class dgen(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):

        X_batch = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        y_batch = self.y[idx * self.batch_size:(idx + 1) * self.batch_size].todense()
        return np.vstack(X_batch),np.vstack(y_batch)

def generate_submission_file(x_test):
	

	threshold=[.22]
	loop=int(194323/batch_size)
	remained=194323%batch_size
	i=0
	test=pd.read_csv('../data/d583b256-d-new_dataset/new_dataset/test.csv')['id']

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


if __name__ == "__main__":
	embedding_matrix, x_train, y_train,x_test, mlb=preprocess_data()
	train(embedding_matrix, x_train, y_train,x_test, mlb)
	print("training completed")

