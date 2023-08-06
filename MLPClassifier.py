import jieba
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
import pandas as pd

md = joblib.load('MLPClassifier')
tt = [' '.join(jieba.cut('这手机真的拉，失望，非常失望'))]
vect = CountVectorizer()
my_try = md.fit(tt)
res = md.predict(my_try)
print(res)


