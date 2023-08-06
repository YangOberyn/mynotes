import pickle
import time

import jieba
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier


data = pd.read_excel('产品评价.xlsx')
rows = []
for i, d in data.iterrows():
    ds = jieba.cut(d['评论'])
    rows.append(' '.join(ds))

# 向量化
vect = CountVectorizer()
tsf = vect.fit_transform(rows)

x = tsf.toarray()
y = data['评价']
print(x.shape)

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.1, random_state=1)

mlp = MLPClassifier(hidden_layer_sizes=(64, 64, 16))

mlp.fit(x_tr, y_tr)

# 预测
y_pred = mlp.predict(x_te)
# 正确率得分
score = accuracy_score(y_pred, y_te)
# score = mlp.score(x_te, y_te)

print(score)

tt = [' '.join(jieba.cut('这手机真的拉，失望，非常失望'))]
my_try = vect.transform(tt)
print(my_try)
res = mlp.predict(my_try)
print(res)

print('模型评价')
print(classification_report(y_pred, y_te))

# 保存训练数据
joblib.dump(mlp, 'MLPClassifier.m')
# 加载训练数据
# joblib.load('MLPClassifier.m')
# 保存向量化
# path = ''
# with open(path, 'wb') as fw:
#     pickle.dump(vect.vocabulary_, fw)
# 加载向量化
# load_vect = CountVectorizer(vocabulary=pickle.load(open('path', 'rb')))

# 贝叶斯模型 BernoulliNB GaussianNB  MultinomialNB
bayes = MultinomialNB()
bayes.fit(x_tr, y_tr)

# 预测
b_pred = bayes.predict(x_te)

b_sc = accuracy_score(y_pred, y_te)
print(b_sc)

