import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
import os
import numpy as np
import collections
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import NuSVC
from sklearn.externals import joblib


#预处理
def preprocessing(text):
    #text=text.decode("utf-8")
    text=text.strip('"')
    tokens=[word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    stops=stopwords.words('english')
    tokens=[token for token in tokens if token not in stops]

    tokens=[token.lower() for token in tokens if len(token)>=3]
    lmtzr=WordNetLemmatizer()
    tokens=[lmtzr.lemmatize(token) for  token in tokens]
    preprocessed_text=' '.join(tokens)
    return preprocessed_text

#读取数据集
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
#先搞训练集
csv_reader=csv.reader(train_file,delimiter='\t')
head_row = next(csv_reader)
print(head_row)
for line in csv_reader:
    train_label.append(line[1])
    train_data.append(preprocessing(line[2]))
train_file.close()
print(train_data[0:5])
print(train_label[0:5])

#按0.7：0.3比例分为训练集和测试集，再将其向量化
dataset_size=len(train_data)
trainset_size=int(round(dataset_size*0.7))
print('dataset_size:',dataset_size,'   trainset_size:',trainset_size)
x_train=np.array([''.join(el) for el in train_data[0:trainset_size]])
y_train=np.array(train_label[0:trainset_size])

print(x_train[0:5])

x_val=np.array(train_data[trainset_size+1:dataset_size])
y_val=np.array(train_label[trainset_size+1:dataset_size])
print("x_val,y_val shape:",x_val.shape,y_val.shape)

#再搞测试集
csv_reader=csv.reader(test_flie,delimiter='\t')
head_row = next(csv_reader)
print(head_row)
for line in csv_reader:
    test_id.append(line[0])
    test_data.append(preprocessing(line[1]))
test_flie.close()
print(test_id[0:5])
print(test_data[0:5])

test_dataset_size=len(test_data)
x_test=np.array(test_data)

vectorizer=TfidfVectorizer(min_df=2,ngram_range=(1,3),stop_words='english',strip_accents='unicode',norm='l2')

X_train=vectorizer.fit_transform(x_train)
X_val=vectorizer.transform(x_val)
X_test=vectorizer.transform(x_test)

feature_names=vectorizer.get_feature_names()

print(feature_names)
print('X_train1-0:',X_train[0])

#LogisticRegression
clf=RidgeClassifier().fit(X_train,y_train)
y_val_pred=clf.predict(X_val)
print('SVM_confusion_matrix:')
cm=confusion_matrix(y_val,y_val_pred)
print(cm)
print('SVM_classification_report:')
print(classification_report(y_val,y_val_pred))

y_test_pred=clf.predict(X_test)
result=zip(test_id,y_test_pred)

f=open('result.csv', 'w',newline='')
csv_writer=csv.writer(f,delimiter=',')
csv_writer.writerow(['id','sentiment'])
csv_writer.writerows(result)

model_dict={}
model_dict['vectorizer']=vectorizer
model_dict['clf']=clf
joblib.dump(model_dict,os.path.join(model_dir,'LR_model.model'))







