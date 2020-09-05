from matplotlib.pyplot import figure
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image
# 导入
market = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
# print(market)
# 删除原始数据左右空格
for i in market.columns:
    market[i] = market[i].str.strip()

items = []
for i in market.index:
    items.extend(market.loc[i].unique())
# 删除NAN
items = [str(i) for i in items if str(i)!= "nan"]
item_count = {}
for i in set(items):
    item_count.update({i:items.count(i)})
# print(item_count)

top_ten=sorted(item_count.items(),key=lambda x:x[1],reverse=True)[:10]
print(top_ten)

# 可视化
count = [i[1] for i in top_ten]
labels = [i[0] for i in top_ten]
plt.pie(count, labels=labels)
plt.show()

# 词云

# 空格分隔符
words='-'.join(items)
word=WordCloud(background_color="white",width=1500,height=1000,font_path='simhei.ttf',).generate(words)
word.to_file('Market_Basket.png')
# 显示词云文件
plt.imshow(word)
plt.axis("off")
plt.show()
# img = Image.open("Market_Basket.png")
# plt.figure(figsize=(15,15))
# plt.axis("off")
# plt.imshow(img)