import os
#print(os.listdir("../input/he-toxic-multilabel"))

# Any results you write to the current directory are saved as output.
import os
import gc
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from keras import backend as K
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.cross_validation import KFold
from keras.preprocessing import text, sequence
from sklearn.preprocessing import MultiLabelBinarizer
import pickle


TRAIN_FILE='data/Dataset-DL4/train.csv'
TEST_FILE='data/Dataset-DL4/test.csv'
EMBEDDING_FILE = 'data/Dataset-DL4/crawl-300d-2M.vec'


def preprocess_data(max_features = 100000,maxlen = 200,embed_size = 300):
	#load and clean data
	train=pd.read_csv(TRAIN_FILE)
	train=train.drop(['id'],axis=1)
	train['tags']=train['tags'].astype(str)
	train['article']=train['article'].str.replace('</p>|<p>|\r|\n|<br>|</p>|<pre>|</pre>|<code>|</code>','')
	train['combined']=train['title']+' '+train['article']
	train.drop(['title','article'],axis=1,inplace=True)
	lst=[x.split(',') for x in train['tags'].str.replace('|',',').tolist()]

	#one hot encode label (multi)
	mlb = MultiLabelBinarizer(sparse_output=True)
	y=mlb.fit_transform(lst)
	with open('multi_label_binarizer.pickle', 'wb') as handle:
		pickle.dump(mlb, handle, protocol=pickle.HIGHEST_PROTOCOL)

	print("binarizer saved")
	del lst
	test=pd.read_csv(TEST_FILE)

	test=test.drop(['id'],axis=1)
	test['article']=test['article'].str.replace('</p>|<p>|\r|\n|<br>|</p>|<pre>|</pre>|<code>|</code>','')

	test['combined']=test['title']+' '+test['article']

	test.drop(['title','article'],axis=1,inplace=True)
	X_train = train["combined"].fillna("fillna").values
	y_train = y#train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
	del y
	X_test = test["combined"].fillna("fillna").values
	tokenizer = text.Tokenizer(num_words=max_features)
	tokenizer.fit_on_texts(list(X_train) + list(X_test))
	with open('tokenizer.pickle', 'wb') as handle:
		pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print("tokenizer saved")
	X_train = tokenizer.texts_to_sequences(X_train)
	X_test = tokenizer.texts_to_sequences(X_test)
	x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
	x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
	gc.collect()
	del X_train, X_test

	embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding="utf8"))

	all_embs = np.stack(embeddings_index.values())
	emb_mean, emb_std = all_embs.mean(), all_embs.std()

	del all_embs 
	gc.collect()

	word_index = tokenizer.word_index
	nb_words = min(max_features, len(word_index))
	embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
	for word, i in word_index.items():
	    if i >= max_features: continue
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
	print('preprocessing done')

	return embedding_matrix, x_train, y_train,x_test, mlb

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

if __name__ == "__main__":
	preprocess_data()
