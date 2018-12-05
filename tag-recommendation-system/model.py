
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


#model
#wrote out all the blocks instead of looping for simplicity
def model_feature(embedding_matrix):
	max_features = 100000
	maxlen = 200
	embed_size = 300
	filter_nr = 64
	filter_size = 3
	max_pool_size = 3
	max_pool_strides = 2
	dense_nr = 256
	spatial_dropout = 0.2
	dense_dropout = 0.5
	train_embed = False
	conv_kern_reg = regularizers.l2(0.00001)
	conv_bias_reg = regularizers.l2(0.00001)

	comment = Input(shape=(maxlen,))
	emb_comment = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=train_embed)(comment)
	emb_comment = SpatialDropout1D(spatial_dropout)(emb_comment)

	block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
	            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(emb_comment)
	block1 = BatchNormalization()(block1)
	block1 = PReLU()(block1)
	block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
	            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1)
	block1 = BatchNormalization()(block1)
	block1 = PReLU()(block1)

	#we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output
	#if you choose filter_nr = embed_size (300 in this case) you don't have to do this part and can add emb_comment directly to block1_output
	resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear', 
	            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(emb_comment)
	resize_emb = PReLU()(resize_emb)
	    
	block1_output = add([block1, resize_emb])
	block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

	block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
	            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1_output)
	block2 = BatchNormalization()(block2)
	block2 = PReLU()(block2)
	block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
	            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2)
	block2 = BatchNormalization()(block2)
	block2 = PReLU()(block2)
	    
	block2_output = add([block2, block1_output])
	block2_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)

	block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
	            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2_output)
	block3 = BatchNormalization()(block3)
	block3 = PReLU()(block3)
	block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
	            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3)
	block3 = BatchNormalization()(block3)
	block3 = PReLU()(block3)
	    
	block3_output = add([block3, block2_output])
	block3_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)

	block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
	            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3_output)
	block4 = BatchNormalization()(block4)
	block4 = PReLU()(block4)
	block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
	            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4)
	block4 = BatchNormalization()(block4)
	block4 = PReLU()(block4)

	block4_output = add([block4, block3_output])
	block4_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block4_output)

	block5 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
	            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4_output)
	block5 = BatchNormalization()(block5)
	block5 = PReLU()(block5)
	block5 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
	            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5)
	block5 = BatchNormalization()(block5)
	block5 = PReLU()(block5)

	block5_output = add([block5, block4_output])
	block5_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block5_output)

	block6 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
	            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5_output)
	block6 = BatchNormalization()(block6)
	block6 = PReLU()(block6)
	block6 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
	            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6)
	block6 = BatchNormalization()(block6)
	block6 = PReLU()(block6)

	block6_output = add([block6, block5_output])
	block6_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block6_output)

	block7 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
	            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6_output)
	block7 = BatchNormalization()(block7)
	block7 = PReLU()(block7)
	block7 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
	            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block7)
	block7 = BatchNormalization()(block7)
	block7 = PReLU()(block7)

	block7_output = add([block7, block6_output])
	output = GlobalMaxPooling1D()(block7_output)

	output = Dense(dense_nr, activation='linear')(output)
	output = BatchNormalization()(output)
	output = PReLU()(output)
	output = Dropout(dense_dropout)(output)
	output = Dense(36321, activation='sigmoid',kernel_initializer=keras.initializers.RandomUniform(minval=-3e-4, maxval=3e-4, seed=None))(output)
	model = Model(comment, output)

	return model

def model_decision(model1):
	output_1=Model(inputs=[model.get_input_at(0)],
                                      outputs=[model.layers[63].output])
	output_2=Model(inputs=[model.get_input_at(0)],
	                                      outputs=[model.layers[65].output])

	x = Dense(512)(output_1.output)
	x=BatchNormalization()(x)
	x= PReLU()(x)
	x = Dense(196)(x)
	x=BatchNormalization()(x)
	x= PReLU()(x)
	theta=Dense(36321)(x)

	logit=Lambda(logits,output_shape=logit_output_shape,name='logits')(output_2.output)

	diff = Lambda(difference,
	                  output_shape=diff_output_shape,name='diff')([theta, logit])
	diff=Activation('sigmoid')(diff)
	model2=Model(inputs=[model.get_input_at(0)],outputs=diff)
	list_layers=[75,74,73,72,70,69,68,67,66,65,64,]
	for layer in range(len(model2.layers)):
	    model2.layers[layer].trainable=False
	for layer in list_layers:
		model2.layers[layer].trainable=True
	return model2


def difference(vectors):
    x,y=vectors
    return (tf.stop_gradient(y)-x)

def diff_output_shape(shapes):
    shape_x,shape_y=shapes
    return shape_x

def logits(vector):
    _epsilon = tf.convert_to_tensor(K.epsilon(), vector.dtype.base_dtype)
    output = tf.clip_by_value(vector, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))
    return tf.stop_gradient(output)

def logit_output_shape(shape):
    return shape


