## 简要介绍一下什么是MVC框架
1. MVC框架 = M(Model) + V(views) + C(controller)
2. model:与数据库相关的模型层，表示业务流程或者状态的处理以及业务规则的制定。模型接受view请求的数据，并返回result。是MVC的核心。
3. views:用户交互界面，代表网页的地址，以及渲染网页等。或者概括为HTML页面。MVC设计模式对于视图的处理仅限于视图上数据的采集和处理，以及用户的请求，而不包括在视图上的业务流程的处理。业务流程的处理交予模型(Model)处理。
4. controller：是属于middleware，负责根据用户从view输入的指令，选取model的数据，然后对其进行相应的操作。

## 基于python的可视化技术都有哪些？你都使用过哪些？

1. Python 里有很多的可视化库，比如 Matplotlib、Seaborn、Bokeh、Plotly、Pyecharts、Mapbox 和 Geoplotlib。其中使用频率最高，最需要掌握的就是 Matplotlib 和 Seaborn。Matplotlib 是 Python 的可视化基础库，作图风格和 MATLAB 类似，所以称为 Matplotlib；Seaborn 是一个基于 Matplotlib 的高级可视化效果库，针对 Matplotlib 做了更高级的封装，让作图变得更加容易。简单来说就是会更漂亮。

2. 还有一种 常用的是用python + Echarts的方式作为可视化的呈现。
3. 另一种是比较对应于文字的可视化呈现---词云，它很美观，常用语NLP领域对于语句进行分词后选出keyWord然后根据每个关键词的重要性去进行展示。
4. 因为现在还在转行准备期，我用到的可视化工具还是matplotlib，用于做机器学习竞赛或者时间序列主题的比赛的时候的数据清洗。

