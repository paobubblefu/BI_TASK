## 什么是监督学习，无监督学习，半监督学习
1. 监督学习：从给定的训练数据集中学习出一个model，当新的数据到来的时候，可以根据这个model预测结果。其训练集需要包括feature和label。
2. 无监督学习：输入数据没有被标记，也没有确定的结果。需要根据样本间的相似性对样本集进行聚类，试图使类内差距最小化，类间差距最大化。它是让计算机自己去学习怎么做事情。
3. 半监督学习：利用少量的标注的样本，和大量未标注样本进行学习的方法。

## K-means中的k值如何选取

利用手肘法去选取K值。就是对n个点的数据集，迭代计算k from 1 to n ，每次聚类完成后计算每个点到其所属的簇中心的距离的平方和。这个和通常会逐渐变小，直到k == n的时候平方和为0，因为那时候每个店都在它所在的簇中心本身了。但在平方和变化过程中，会出现一个拐点，拐点后下降率突然变缓，那个点就是K指。

## 随机森林采用了bagging集成学习，bagging指的是什么

bagging就是训练k个独立的基学习器，对于每个基学习器的结果进行结合（加权或者多数投票）来获得一个强学习器。

bagging算法特点是各个弱学习器之间没有依赖关系，可以并行拟合。加快计算速度。

bagging的步骤为：

    A）从原始样本集中抽取训练集。每轮从原始样本集中使用Bootstraping的方法抽取n个训练样本（在训练集中，有些样本可能被多次抽取到，而有些样本可能一次都没有被抽中）。共进行k轮抽取，得到k个训练集。（k个训练集之间是相互独立的）
    B）每次使用一个训练集得到一个模型，k个训练集共得到k个模型。（注：这里并没有具体的分类算法或回归方法，我们可以根据具体问题采用不同的分类或回归方法，如决策树、感知器等）
    C）对分类问题：将上步得到的k个模型采用投票的方式得到分类结果；对回归问题，计算上述模型的均值作为最后的结果。（所有模型的重要性相同）


## 主动学习和半监督学习的区别是什么
对于主动学习：
    如果机器可以自己选择学习的样本，它可以使用较少的训练取得更好的效果
需要人工介入，模型主动向worker提供数据

对于半监督学习：
指在训练数据十分稀少的情况下，利用没有标签的数据，提高模型效果的方法



