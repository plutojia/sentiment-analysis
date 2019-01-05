import numpy as np
import os
import csv
from keras import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers import *
import keras
from keras import Model
from keras.callbacks import *


Embedding_DIR='your word embedding data'
file_path='your data'
testflie_path='your data'
current_dir = os.getcwd()
model_dir=os.path.join(current_dir, 'model')

train_file=open(file_path,'r',encoding='ISO-8859-1')
test_flie=open(testflie_path,'r',encoding='ISO-8859-1')
train_data=[]
train_label=[]
test_data=[]
test_id=[]


csv_reader=csv.reader(train_file,delimiter=',')
head_row = next(csv_reader)
print(head_row)
for line in csv_reader:
    train_label.append(line[1])
    train_data.append(line[2])
train_file.close()

#preprocess
MAX_NB_WORDS=55000
MAX_SEQUENCE_LENGTH=500
VALIDATION_SPLIT=0.3
EMBEDDING_DIM=300

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(train_data)
sequences = tokenizer.texts_to_sequences(train_data)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
train_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
train_label = keras.utils.to_categorical(np.asarray(train_label))
print('Shape of data tensor:', train_data.shape)
print('Shape of train_label tensor:', train_label.shape)

# split the data into a training set and a validation set
indices = np.arange(train_data.shape[0])
np.random.shuffle(indices)
train_data = train_data[indices]
train_label = train_label[indices]
nb_validation_samples = int(VALIDATION_SPLIT * train_data.shape[0])

x_train = train_data[0:nb_validation_samples]
y_train = train_label[0:nb_validation_samples]
x_val = train_data[-nb_validation_samples:]
y_val = train_label[-nb_validation_samples:]


csv_reader=csv.reader(test_flie,delimiter=',')
head_row = next(csv_reader)
print(head_row)
for line in csv_reader:
    test_id.append(line[0])
    test_data.append(line[1])
test_flie.close()
test_sequences = tokenizer.texts_to_sequences(test_data)

test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', test_data.shape)
x_test=test_data

#Embedding layer
embeddings_index = {}
f = open(os.path.join(Embedding_DIR, 'your Embedding_file '),'r',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs_str=[float(s) for s in values[1:]]
    coefs = np.asarray(coefs_str, dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

#network
pool_output = []
pool_output2 = []
pool_output3=[]
kernel_sizes = [3, 4, 5]
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)


for kernel_size in kernel_sizes:
    c = Conv1D(filters=100, kernel_size=kernel_size, strides=1, padding='valid')(embedded_sequences)
    p = MaxPool1D(pool_size=int(c.shape[1]), strides=2)(c)
    pool_output.append(p)

concatenated = concatenate([p for p in pool_output])
print("pool_output.shape: %s" % str(concatenated.shape))  # (?, 1, 6)
x_flatten = Flatten()(concatenated)

dropouted_1 = Dropout(0.5)(x_flatten)

preds = Dense(2, activation='softmax')(dropouted_1)

'''
def Res(x,filters,kernel_size):
    x1 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1,padding='same',activation='relu')(x)
    x2 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same',activation='relu')(x1)
    x_out=Add()([x,x2])
    return x_out

dropout_1=Dropout(0.3)(embedded_sequences)
x = Conv1D(filters=200, kernel_size=200, strides=1, padding='same',activation='relu')(dropout_1)
                                                #1500
res_1=Res(x,200,3)
res_1=MaxPooling1D(3,strides=2)(res_1)          #750

res_2=Res(res_1,200,3)
res_2=MaxPooling1D(3,strides=2)(res_2)          #375

res_3=Res(res_2,200,3)
res_3=MaxPooling1D(3,strides=2)(res_3)          #187

res_4=Res(res_3,200,3)
res_4=MaxPooling1D(3,strides=2)(res_4)          #93

res_5=Res(res_4,200,3)
res_5=MaxPooling1D(3,strides=2)(res_5)          #93

x_flatten = Flatten()(res_5)
dropout_2=Dropout(0.4)(x_flatten)
preds = Dense(2, activation='softmax')(dropout_2)
'''

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

checkpointer = ModelCheckpoint(filepath=os.path.join(current_dir,"new_model","checkpoint-{epoch:02d}e-val_acc_{val_acc:.2f}.h5"),
                               save_best_only=False, verbose=1, period=1)

model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=11, batch_size=64,callbacks=[checkpointer])

test_preds=model.predict(x_test,verbose=1)
test_preds=np.argmax(test_preds,axis=1)
print(len(test_preds))

result=zip(test_id,test_preds)

f=open('result.csv', 'w',newline='')
csv_writer=csv.writer(f,delimiter=',')
csv_writer.writerow(['id','sentiment'])
csv_writer.writerows(result)
f.close()

model.save(os.path.join(model_dir,'textcnn_model.h5'),overwrite=True)
