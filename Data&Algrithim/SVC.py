
import numpy as np
from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split,cross_val_score, KFold
import joblib


# 1.读取数据集
df=pd.read_excel(io=r"C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\原始数据\\features20231222增强gauss噪声改进后.xlsx")
# print(df.values)
#print(df.values[0,0])    #通过df.valued[]的方式引用数据
# print(type(df.values[0,1]))
#row = len(df.values)
#print(row)  # 输出行数，（数据的组数）


# 2.获取特征和标签，划分训练集、测试集
features_num = 14
features = df.values[:, 0:features_num]
labels = df.values[:, features_num]
train_data, test_data, train_label, test_label = train_test_split(features, labels, random_state=10, train_size=0.8)    # 训练集，测试集占比
                                                                    # sklearn.model_selection.


# 3.训练svm分类器
# svm.SVC函数参数：
classifier = svm.SVC(C=5, kernel='rbf', gamma=5, decision_function_shape='ovr', probability=True)  # ovr和ovo,根据情况选择
import time
start = time.perf_counter()
classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先
end = time.perf_counter()
print("时间：", str(end-start))

'''
kfold = KFold(n_splits=10)
# 执行十折交叉验证
scores = cross_val_score(classifier, features,labels, cv=kfold)  # train_data, train_label   test_data, test_label
# 输出每折的准确率
for i, score in enumerate(scores):
    print("Fold {}: {:.4f}".format(i+1, score))
# 输出平均准确率
print("Average Accuracy: {:.4f}".format(scores.mean()))
#save model
'''
joblib.dump(classifier, r'C:\Users\jiang\Desktop\all_data\20240106SI\SVC\SVC.pkl')

# 4.计算svc分类器的准确率
print("训练集：", classifier.score(train_data, train_label))
print("测试集：", classifier.score(test_data, test_label))
pre_train_label = classifier.predict(train_data)  # 训练集的预测标签
pre_test_label = classifier.predict(test_data)  # 测试集的预测标签

# 查看决策函数
print('train_decision_function:\n', classifier.decision_function(test_data))
#print('predict_result:\n', classifier.predict(train_data))


# 5.可视化
disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        test_data,
        test_label,
        cmap=plt.cm.Blues,
        normalize="true",
    )
disp.ax_.set_title('Support Vector Classifier Accuracy')
plt.savefig(r"C:\Users\jiang\Desktop\all_data\20240106SI\SVC\SVC confusion matrix normal.png", dpi=600)    #保存图片一定要在plt.show()之前
plt.savefig(r'C:\Users\jiang\Desktop\all_data\20240106SI\SVC\SVC confusion matrix normal.svg')
plt.show()


#绘制：测试集的混淆矩阵(confusion matrix)
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(test_label, pre_test_label)
plt.imshow(confusion, cmap=plt.cm.Blues)
# 热度图，后面是指定的颜色块，可设置其他的不同颜色
# ticks 坐标轴的坐标点
# label 坐标轴标签说明
indices = range(len(confusion))
# 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
# plt.xticks(indices, [0, 1, 2])
# plt.yticks(indices, [0, 1, 2])
plt.xticks(indices, [ 'K', 'N', 'P', 'health','white'])
plt.yticks(indices, [ 'K', 'N', 'P', 'health','white'])
plt.colorbar()
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.title('Support Vector Classifier Accuracy')
# plt.rcParams两行是用于解决标签不能显示汉字的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 显示数据
for first_index in range(len(confusion)):  # 第几行
    for second_index in range(len(confusion[first_index])):  # 第几列
        plt.text(first_index, second_index, confusion[second_index][first_index])

plt.savefig(r'C:\Users\jiang\Desktop\all_data\20240106SI\SVC\SVC confusion matrix.png', dpi=600)    #保存图片一定要在plt.show()之前
plt.show()



# 模型参数固定，精度随着训练样本增加的变化曲线
score = []
n = 160
t = int(800/n)
print(t)
for i in range(1, n+1):
    classifier.fit(train_data[0:t*i-1, :], train_label[0:t*i-1])
    score_test = classifier.score(test_data, test_label)
    score.append(score_test)
print(score)
plt.plot(range(1,n+1), score)
plt.ylabel('Accuracy')
plt.xlabel('scale of train data')
plt.ylim(0,1)
plt.title('DT Test Accuracy')
plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\SVC\\160accuracy_data_scale.png', dpi = 600)    #保存图片一定要在plt.show()之前
plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\SVC\\SVC accuracy_data_scale.eps', dpi = 600)    #保存图片一定要在plt.show()之前
plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\SVC\\SVC accuracy_data_scale.svg')
plt.show()

score_to_excel = pd.DataFrame(data=score)
score_to_excel.to_excel(r'C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\SVC\\SVCscore.xlsx')



from sklearn.metrics import f1_score
pre_test_label = classifier.predict(test_data)
print("F1_score:", f1_score(test_label, pre_test_label, average='macro'))


## 可视化每个类别的ROC曲线
from sklearn.metrics import roc_curve,roc_curve, roc_auc_score

lable_names = ["K","N","P","health","white"]
colors = ["r","b","g","m","k",]
linestyles =["-", "--", "-.", ":", "-"]
pre_score = classifier.predict_proba(test_data)
#print(pre_score)
#print(pre_test_label)
#print(test_label)

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
inset_ax = fig.add_axes([0.3, 0.45, 0.4, 0.4],facecolor="white")
for ii, color in zip(range(5), colors):
    ## 计算绘制ROC曲线的取值
    fpr_ii, tpr_ii, _ = roc_curve(Y_onehot[:, ii], pre_score[:, ii])
    ## 局部放大图
    inset_ax.plot(fpr_ii, tpr_ii,color = color,linewidth = 2,
                  linestyle = linestyles[ii])
    inset_ax.set_xlim([-0.01,0.06])
    inset_ax.set_ylim([0.8,1.01])
    inset_ax.grid()
plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\SVC\\SVC ROC-AUC.svg')
plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240106SI\\SVC\\SVC ROC-AUC.png')
plt.show()