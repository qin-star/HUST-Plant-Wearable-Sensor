

import pandas as pd
from sklearn import tree
from sklearn.tree import plot_tree # 树图
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import joblib
import time
import numpy as np

# 读取数据
df=pd.read_excel(io=r"C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\原始数据\\features20231222增强gauss噪声改进后.xlsx")
# print(df.values)
#print(df.values[0,0])    #通过df.valued[]的方式引用数据
# print(type(df.values[0,1]))
row = len(df.values)
#print(row)  # 输出行数，（数据的组数）

# 获取特征和标签，划分训练集、测试集
features_num = 14
features = df.values[:, 0:features_num]
labels = df.values[:, features_num]
train_data, test_data, train_label, test_label = train_test_split(features, labels, random_state=12, train_size=0.8)    # 训练集，测试集占比
                                                                    # sklearn.model_selection.
# print(labels)

# 决策树建模
# 具体的参数调节（信息熵/基尼系数、剪枝参数等）
# 参考：https://blog.csdn.net/qq_47180202/article/details/120427937
# '''
model = tree.DecisionTreeClassifier(max_depth=5
                                    , criterion="entropy"    # entropy效果好
                                    , random_state=3           # 这个参数影响不大
                                    , splitter="best"
                                    , min_samples_leaf=1
                                    , max_features = 14
                                  # , min_impurity_decrease = 0.01     # 该参数默认，效果好
                                    , min_samples_split = 16          #
                                    )
start = time.perf_counter()
model = model.fit(train_data, train_label)
end = time.perf_counter()
print("时间：", str(end-start))

#save model
#joblib.dump(model, r"C:\Users\jiang\Desktop\all_data\20240106SI\DT\Decision-tree.pkl")
'''
# 创建十折交叉验证对象
from sklearn.model_selection import train_test_split,cross_val_score, KFold
kfold = KFold(n_splits=10)
# 执行十折交叉验证
scores = cross_val_score(model, features,labels, cv=kfold)  # train_data, train_label   test_data, test_label
# 输出每折的准确率
for i, score in enumerate(scores):
    print("Fold {}: {:.4f}".format(i+1, score))
# 输出平均准确率
print("Average Accuracy: {:.4f}".format(scores.mean()))
'''


# 查看模型在测试集的分类效果
score_test = model.score(test_data, test_label)
print('测试集精度: ', score_test)
score_train = model.score(train_data, train_label)
print('训练集精度: ', score_train)


'''
pipeline = Pipeline([("classifier", model)])
pipeline.fit(train_data, train_label)
from nyoka import xgboost_to_pmml
xgboost_to_pmml(pipeline, "1",  "2", '/DecisionTreeClassifier.pmml')
'''


# 读取0919的数据，用于验证.(结果：精度只有0.43，说明新加入的数据和之前的数据差异很大，大概率是光线问题，因此不建议把0919的数据汇总，进行训练)

# 决策树可视化

feature_names = df.columns.unique().tolist()
print(feature_names)
label_names = df["labels"].unique().tolist()
print(label_names)
plt.figure(figsize=(30,20))                #设置画布大小（单位为英寸）
# plt.rcParams两行是用于解决标签不能显示汉字的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plot_tree(model,
          feature_names = feature_names[:features_num],
          class_names = label_names,
          filled = True,                    #给节点填充颜色
          rounded = True,                   #节点方框变成圆角
          fontsize=11)                      #节点中文本的字体大小

#plt.savefig(r"C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\DT\\tree_visualization.png")    #保存图片一定要在plt.show()之前
plt.show()



#  绘制：测试集的归一化的混淆矩阵(normalize confusion matrix)
disp = ConfusionMatrixDisplay.from_estimator(
        model,
        test_data,
        test_label,
        #display_labels=[ 'K', 'N', 'P', 'health','white'],
        cmap=plt.cm.Blues,
        normalize="true",
    )
disp.ax_.set_title('Decision Tree Accuracy')
#plt.savefig(r"C:\Users\jiang\Desktop\all_data\20240106SI\DT\Decision Tree confusion matrix normal.png", dpi = 600)    #保存图片一定要在plt.show()之前
#plt.savefig(r'C:\Users\jiang\Desktop\all_data\20240106SI\DT\Decision Tree confusion matrix normal.svg')
plt.savefig(r'C:\Users\jiang\Desktop\all_data\20240106SI\DT\Decision Tree confusion matrix normal.eps')
plt.show()
#  绘制：测试集的混淆矩阵(confusion matrix)
pre_test_label = model.predict(test_data)
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(test_label, pre_test_label)
# 热度图，后面是指定的颜色块，可设置其他的不同颜色
plt.imshow(confusion, cmap=plt.cm.Blues)
# ticks 坐标轴的坐标点
# label 坐标轴标签说明
indices = range(len(confusion))
plt.xticks(indices, [ 'K', 'N', 'P', 'health','white'])
plt.yticks(indices, [ 'K', 'N', 'P', 'health','white'])
plt.colorbar()
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.title('DecisionTree Accuracy')
# plt.rcParams两行是用于解决标签不能显示汉字的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 显示数据
for first_index in range(len(confusion)):  # 第几行
    for second_index in range(len(confusion[first_index])):  # 第几列
        plt.text(first_index, second_index, confusion[second_index][first_index])
        
#plt.savefig(r"C:\Users\jiang\Desktop\all_data\20240106SI\DT\Decision Tree confusion matrix.png", dpi = 600)    #保存图片一定要在plt.show()之前
plt.show()




'''
# 优化超参数
# 1、手动优化
test = []
train = []
n = 30
for i in range(n):
    model = tree.DecisionTreeClassifier(max_depth=5
                                    , criterion="entropy"
                                    , random_state=1           # 这个参数影响不大
                                    , splitter="best"
                                    , min_samples_leaf=1
                                    , max_features = 5
                                    , min_impurity_decrease = 0.001   # 越小越好
                                    , min_samples_split = 9     # 该参数必须大于等于2

                                    )
    model = model.fit(train_data, train_label)
    score = model.score(test_data, test_label)
    test.append(score)
    score = model.score(train_data, train_label)
    train.append(score)

plt.plot(range(1,n+1),test,color="red",label="test")
plt.plot(range(1,n+1),train,color="blue",label="train")
plt.legend()
plt.show()



# 2、网格优化
from sklearn.model_selection import GridSearchCV

params = {'max_depth': range(3, 10, 1),
          'min_samples_split': range(2, 20, 1),
         #  'random_state': range(1, 50, 1),
         # 'criterion': ["entropy","gini"],  # 比gini系数好一些
          'min_samples_leaf': range(1, 20, 1),
          }
gs = GridSearchCV(model, params, cv=5)  # cv=5,交叉验证的折数
gs.fit(train_data, train_label)
# 查验优化后的超参数配置
print(gs.best_score_)
print(gs.best_params_)
#
'''


# 模型参数固定，精度随着训练样本增加的变化曲线
score = []
n = 160
t = int(800/n)
print(t)
for i in range(1, n+1):
    model.fit(train_data[0:t*i-1, :], train_label[0:t*i-1])
    score_test = model.score(test_data, test_label)
    score.append(score_test)
print(score)
plt.plot(range(1,n+1), score)
plt.ylabel('Accuracy')
plt.xlabel('scale of train data')
plt.ylim(0,1)
plt.title('DT Test Accuracy')
#plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\DT\\160accuracy_data_scale.png', dpi = 600)    #保存图片一定要在plt.show()之前
plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\DT\\160accuracy_data_scale.eps', dpi = 600)    #保存图片一定要在plt.show()之前
#plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\DT\\160accuracy_data_scale.svg')
plt.show()

score_to_excel = pd.DataFrame(data=score)
#score_to_excel.to_excel(r'C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\DT\\DTscore.xlsx')


# F1-score
from sklearn.metrics import f1_score
pre_test_label = model.predict(test_data)
print("F1_score:", f1_score(test_label, pre_test_label, average='macro'))

## 可视化每个类别的ROC曲线
from sklearn.metrics import roc_curve,roc_curve, roc_auc_score

lable_names = ["K","N","P","health","white"]
colors = ["r","b","g","m","k",]
linestyles =["-", "--", "-.", ":", "-"]
pre_score = model.predict_proba(test_data)
print(pre_score)
print(pre_test_label)
print(test_label)

# 画多分类的ROC曲线，就必须把预测的标签转换为[0，0，1，0，0]这种one_hot编码的形式
pre_test = np.zeros((200,5))
for i in range(len(test_label)):
    index = np.argmax(pre_score[i,:])
    pre_test[i,index] = 1
print(pre_test)
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(test_label)
Y_onehot = np_utils.to_categorical(Y_encoded)


fig  = plt.figure(figsize=(8,7))
for ii, color in zip(range(5), colors):
    ## 计算绘制ROC曲线的取值
    fpr_ii, tpr_ii, _ = roc_curve(Y_onehot[:, ii], pre_score[:, ii])
    auc = roc_auc_score(Y_onehot[:, ii], pre_score[:, ii])
    plt.plot(fpr_ii, tpr_ii,color = color,linewidth = 2,
             linestyle = linestyles[ii],
             label = "class:"+lable_names[ii]+"    AUC:"+str(auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("假正率")
plt.ylabel("真正率")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.legend()
plt.title("每个类别的ROC曲线")
## 添加局部放大图
inset_ax = fig.add_axes([0.3, 0.4, 0.4, 0.4],facecolor="white")
for ii, color in zip(range(5), colors):
    ## 计算绘制ROC曲线的取值
    fpr_ii, tpr_ii, _ = roc_curve(Y_onehot[:, ii], pre_score[:, ii])
    ## 局部放大图
    inset_ax.plot(fpr_ii, tpr_ii,color = color,linewidth = 2,
                  linestyle = linestyles[ii])
    inset_ax.set_xlim([-0.01,0.06])
    inset_ax.set_ylim([0.8,1.01])
    inset_ax.grid()
plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240115SI\\DT\\DT ROC-AUC.svg')
plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240115SI\\DT\\DT ROC-AUC.png')
plt.show()



