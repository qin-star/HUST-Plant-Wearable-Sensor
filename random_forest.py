from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib
import warnings
import numpy as np
warnings.filterwarnings("ignore")


# 读取数据
df=pd.read_excel(io=r"C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\原始数据\\features20231222增强gauss噪声改进后.xlsx")
# print(df.values)
print(df.values[0,0])    #通过df.valued[]的方式引用数据
# print(type(df.values[0,1]))
row = len(df.values)
print(row)  # 输出行数，（数据的组数）


# 获取特征和标签，划分训练集、测试集
features_num = 14
features = df.values[:, 0:features_num]
labels = df.values[:, features_num]
train_data, test_data, train_label, test_label = train_test_split(features, labels, random_state=1, train_size=0.8)    # 训练集，测试集占比
                                                                    # sklearn.model_selection.

# 建模
model = RandomForestClassifier(n_estimators=17             # 默认为10
                               , min_samples_split=3
                               , random_state=0
                               , oob_score=True          # 默认为False。推荐使用True，因为这样可以反映模型拟合后的泛化能力。
                               , criterion="entropy"     # 比gini系数好一些
                               , max_depth=6
)
# n_estimators：弱学习器最大迭代次数，即弱学习器的个数。一般而言，该值太小易发生欠拟合，太大则成本增加且效果不明显。一般选取适中的数值，默认为10。


# 训练
model = model.fit(train_data, train_label)
# 查看模型在训练集、测试集的分类效果
score_test = model.score(test_data, test_label)
print('测试集精度: ', score_test)
score_train = model.score(train_data, train_label)
print('训练集精度: ', score_train)

#save model
joblib.dump(model, "C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\RF\\Random Forest.pkl")


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


# '''
#  绘制：测试集的归一化的混淆矩阵(normalize confusion matrix)
disp = ConfusionMatrixDisplay.from_estimator(
        model,
        test_data,
        test_label,
        cmap=plt.cm.Blues,
        normalize="true",
    )
disp.ax_.set_title('Random Forest Accuracy')
plt.savefig(r"C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\RF\\Random Forest confusion matrix normal.png")    #保存图片一定要在plt.show()之前
plt.show()

#   绘制：测试集的混淆矩阵(confusion matrix)
pre_test_label = model.predict(test_data)
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(test_label, pre_test_label)
# 热度图，后面是指定的颜色块，可设置其他的不同颜色
plt.imshow(confusion, cmap=plt.cm.Oranges)
# ticks 坐标轴的坐标点
# label 坐标轴标签说明
indices = range(len(confusion))
plt.xticks(indices, [ 'K', 'N', 'P', 'health', 'white'])
plt.yticks(indices, [ 'K', 'N', 'P', 'health', 'white'])
plt.colorbar()
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.title('Random Forest Accuracy')
# plt.rcParams两行是用于解决标签不能显示汉字的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 显示数据
for first_index in range(len(confusion)):  # 第几行
    for second_index in range(len(confusion[first_index])):  # 第几列
        plt.text(first_index, second_index, confusion[second_index][first_index])
plt.savefig(r"C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\RF\\Random Forest confusion matrix.png")    #保存图片一定要在plt.show()之前
plt.show()


# 超参优化：1、网格优化）  2、随机搜索
'''
# 1、网格优化
from sklearn.model_selection import GridSearchCV

params = {'n_estimators': range(3, 20 , 1),
          'min_samples_split': range(2, 20, 1),
         #  'random_state': range(1, 50, 1),
          'oob_score': [True],  # 默认为False。推荐使用True，因为这样可以反映模型拟合后的泛化能力。
         'criterion': ["entropy", "gini"],  # 比gini系数好一些
          'max_depth': range(2, 8, 1)
          }
gs = GridSearchCV(model, params, cv=5)  # cv=5,交叉验证的折数
gs.fit(train_data, train_label)
# 查验优化后的超参数配置
print(gs.best_score_)
print(gs.best_params_)
# 优化后参数记录




# 2、随机搜索
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
# 给定参数搜索范围：list or distribution
param_dist = {"max_depth": sp_randint(3, 30),  # 给定list
              "max_features": sp_randint(1, 30),  # 给定distribution
              "min_samples_split": sp_randint(2, 100),  # 给定distribution
              "bootstrap": [True, False],  # 给定list
              "criterion": ["gini", "entropy"],
              "oob_score": [True, False],
              "random_state": sp_randint(1, 30)
              }  # 给定list
# 用RandomSearch+CV选取超参数
n_iter_search = 20
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter_search, cv=4)
random_search.fit(train_data, train_label)
print(random_search.best_score_)
print(random_search.best_params_)
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
#plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\RF\\160accuracy_data_scale.png', dpi = 600)    #保存图片一定要在plt.show()之前
plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\RF\\160accuracy_data_scale.eps', dpi = 600)    #保存图片一定要在plt.show()之前
#plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\RF\\160accuracy_data_scale.svg')
plt.show()

score_to_excel = pd.DataFrame(data=score)
#score_to_excel.to_excel(r'C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\RF\\RFscore.xlsx')



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
plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\RF\\RF ROC-AUC.svg')
plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\RF\\RF ROC-AUC.png')
plt.show()
