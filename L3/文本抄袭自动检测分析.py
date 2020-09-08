'''
Step1，数据加载 加载sqlResult.csv及停用词chinese_stopwords.txt
Step2，数据预处理
    1）数据清洗，针对content字段为空的情况，进行dropna
    2）分词，使用jieba进行分词
    3）将处理好的分词保存到 corpus.pkl，方便下次调用
    4）数据集切分，70%训练集，30%测试集
Step3，提取文本特征TF-IDF
Step4，预测文章风格是否和自己一致 使用分类模型（比如MultinomialNB），对于文本的特征（比如TF-IDF）和 label（是否为新华社）进行训练

Step5，找到可能Copy的文章，即预测label=1，但实际label=0
Step6，根据模型预测的结果来对全量文本进行比对，如果数量很大，我们可以先用k-means进行聚类降维，比如k=25种聚类 Step7，找到一篇可能的Copy文章，从相同label中，找到对应新华社的文章，并按照TF-IDF相似度矩阵，从大到小排序，取Top10
Step8，使用编辑距离editdistance，计算两篇文章的距离

Step9，精细比对，对于疑似文章与原文进行逐句比对，即计算每个句子的编辑距离editdistance

'''


import pandas as pd 
import numpy as np
import jieba

# 导入数据
news = pd.read_csv('sqlResult.csv', encoding = 'gb18030')
news = news.dropna(subset=['content'])
# 载入停用词
with open('chinese_stopwords.txt', encoding='utf-8') as file:
    stopwords = [i[:-1] for i in file.readlines()]

# 分词

def split_text(text):
    text = text.replace(' ', '').replace('\n', '')
    text2 = jieba.cut(text.strip())
    # 去停用词
    res = ' '.join([w for w in text2 if w not in stopwords])
    return res

# 所有文本进行分词

corpus = list(map(split_text, [str(i) for i in news.content]))
print(corpus[0])
print(len(corpus))

#保存到文件
import pickle
with open('corpus.pkl', 'wb') as file:
    pickle.dump(corpus, file)
 
#计算corpus中的TF-IDF矩阵
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
countvectorizer = CountVectorizer(encoding='gb18030', min_df=0.015)
tfidftransformer = TfidfTransformer()
#先做TF，再做IDF => TF-IDF
countvector = countvectorizer.fit_transform(corpus)
tfidf = tfidftransformer.fit_transform(countvector)
 
#标记是否自己的新闻
label = list(map(lambda source:1 if '新华' in str(source) else 0, news.source))
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf.toarray(), label, test_size=0.3)
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf.toarray(), label, test_size=0.3)
 
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_predict= clf.predict(X_test)
 
import numpy as np
#使用模型检测抄袭新闻，预测风格
prediction = clf.predict(tfidf.toarray())
labels = np.array(label)
compare_news_index = pd.DataFrame({'prediction': prediction, 'labels': labels})
copy_news_index = compare_news_index[(compare_news_index['prediction']==1) & (compare_news_index['labels']==0)]
#实际为新华社的新闻
xinhuashe_news_index = compare_news_index[(compare_news_index['labels']==1)].index
 
print('可能为copy的新闻条数', len(copy_news_index))

#使用Kmeans对文章进行聚类
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
normalizer = Normalizer()
scaled_array= normalizer.fit_transform(tfidf.toarray())
 
kmeans = KMeans(n_clusters=10)
k_labels = kmeans.fit_predict(scaled_array)
 
#创建id_class
id_class = {index:class_ for index, class_ in enumerate(k_labels)}
from collections import defaultdict
class_id = defaultdict(set)
for index, class_ in id_class.items():
    #只统计新华社发布的class_id
    if index in xinhuashe_news_index.tolist():
        class_id[class_].add(index)
 
from sklearn.metrics.pairwise import cosine_similarity
#查找相似文本
def find_similar_text(cpindex, top=10):
    #只在新华社发布的文章中找
    dist_dict = {i:cosine_similarity(tfidf[cpindex], tfidf[i]) for i in class_id[id_class[cpindex]]}
    #从大到小进行排序
    return sorted(dist_dict.items(), key=lambda x:x[1], reverse=True)[:top]
 
cpindex = 3352
similar_list = find_similar_text(cpindex)
print(similar_list)
print('怀疑抄袭\n', news.iloc[cpindex].content)
#找一篇相似的原文
similar2 = similar_list[0][0]
print('相似原文：\n', news.iloc[similar2].content)