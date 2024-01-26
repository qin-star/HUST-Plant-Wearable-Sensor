
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils, plot_model
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 载入数据.数据集里9列：前8个chan+NIR.，没有clear。因为clear都为1
df = pd.read_excel(r"C:\Users\jiang\Desktop\all_data\20240116CNN_concat\原始数据\20231222数据增强gauss噪声改进后_for_CNN.xlsx")
X = np.expand_dims(df.values[:, 0:18].astype(float), axis=2)
Y = df.values[:, 18]
Y_label = Y
print(len(Y))
print(len(X))


# 类别编码为数字  (就是把标签0-4，变成了这样表示（one-hot类型）：以第1类为例：[1,0,0,0,0]，第2类：[0,1,0,0,0]),以此类推
# 但是问题来了，实际上神经网络预测的是每一个类别的概率，最终是选择概率最大的那个类别作为结果。因此在混淆矩阵的绘制过程，需要将真实的标签值再转化成普通的表示方法，才能和预测出来的标签值搭配，
# 所以补充了代码：Y_test = np.argmax(Y_test, 1)，即取预测概率最大的那种类作为结果，该代码约在121行
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)
Y_onehot = np_utils.to_categorical(Y_encoded)
print(Y_onehot.shape)
#print(Y_onehot)


# 划分训练集，测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.2, random_state=1)
print(X_train.shape)


# 定义神经网络
# 两个网络并联
# 定义模型1
model_1 = Sequential()
model_1.add(Conv1D(32, 3, strides=1, input_shape=(9, 1), kernel_initializer='random_normal'))
model_1.add(MaxPooling1D(2))
model_1.add(Flatten())
input1 = model_1.input
print(input1)
output1 = model_1.output
print(output1)

# 定义模型2
model_2 = Sequential()
model_2.add(Conv1D(32, 3, strides=1, input_shape=(9, 1), kernel_initializer='random_normal'))
model_2.add(MaxPooling1D(2))
model_2.add(Flatten())
input2 = model_2.input
print(input2)
output2 = model_2.output
print(output2)

# 合并两个模型  model concat
#concatenated = keras.backend.concatenate([input1, input2], axis=1)
concatenated = keras.layers.concatenate([output1, output2], axis=1)
model = Dense(100, activation='relu')(concatenated)
final_output = Dense(5, activation='softmax')(model)
model = keras.Model(inputs=[input1, input2], outputs=final_output)


print(model.summary())  # 打印模型的摘要信息

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # 编译模型，配置损失函数、优化器和评估指标


#### 训练和测试：这里的输入输出注意，特别是输入
#model.fit(x=[X_train[:,0:9,:], X_train[:,9:18,:]], y=Y_train, epochs=100,batch_size=20, verbose=1)
#model.evaluate(x=[X_test[:,0:9,:], X_test[:,9:18,:]], y=Y_test, verbose=1)


# 保存模型
#model_json = model.to_json()
#with open(r"C:\Users\jiang\Desktop\all_data\20240116CNN_concat\CNN\CNN_concat_model.json", 'w') as json_file:
#    json_file.write(model_json)  # 权重不在json中,只保存网络结构
#model.save_weights(r"C:\Users\jiang\Desktop\all_data\20240116CNN_concat\CNN\CNN_concat_model.h5")


'''
# 创建十折交叉验证对象
from sklearn.model_selection import train_test_split,cross_val_score, KFold
kfold = KFold(n_splits=10)
# 执行十折交叉验证
scores = cross_val_score(model, [X[:,0:9,:], X[:,9:18,:]], Y_onehot, cv=kfold)  # train_data, train_label   test_data, test_label
# 输出每折的准确率
for i, score in enumerate(scores):
    print("Fold {}: {:.4f}".format(i+1, score))
# 输出平均准确率
print("Average Accuracy: {:.4f}".format(scores.mean()))
'''

# 加载模型用做预测
json_file = open(r"C:\Users\jiang\Desktop\all_data\20240116CNN_concat\CNN\CNN_concat_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(r"C:\Users\jiang\Desktop\all_data\20240116CNN_concat\CNN\CNN_concat_model.h5")
print("loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 分类准确率
print("The accuracy of in test data :")
scores = loaded_model.evaluate([X_test[:,0:9,:], X_test[:,9:18,:]], Y_test, verbose=0)
print('%s: %.2f%%' % (loaded_model.metrics_names[1], scores[1] * 100))
# 输出预测类别



# 测试集的混淆矩阵
predicted_label = loaded_model.predict([X_test[:,0:9,:], X_test[:,9:18,:]])   # 得到的为各类的概率值
#print(predicted_label)
predicted_label = np.argmax(predicted_label, 1)    # 返回最大的概率值对应的索引号，0或1或2或3或4，也就对应不同的类别
#print(predicted_label)
Y_test_no_encoder = np.argmax(Y_test, 1)
#print(Y_test)
confusion = confusion_matrix(Y_test_no_encoder, predicted_label)
# 热度图，后面是指定的颜色块，可设置其他的不同颜色
plt.imshow(confusion, cmap=plt.cm.Blues)
# ticks 坐标轴的坐标点
# label 坐标轴标签说明
indices = range(len(confusion))
# 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
plt.xticks(indices, ['K', 'N', 'P', 'health', 'white'])
plt.yticks(indices, ['K', 'N', 'P', 'health', 'white'])
plt.colorbar()
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.title('CNN Accuracy')
# plt.rcParams两行是用于解决标签不能显示汉字的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 显示数据
for first_index in range(len(confusion)):  # 第几行
    for second_index in range(len(confusion[first_index])):  # 第几列
        plt.text(first_index, second_index, confusion[second_index][first_index])
plt.savefig(r'C:\Users\jiang\Desktop\all_data\20240116CNN_concat\CNN\CNN confusion matrix.png', dpi=600)    #保存图片一定要在plt.show()之前
plt.savefig(r'C:\Users\jiang\Desktop\all_data\20240116CNN_concat\CNN\CNN confusion matrix.svg')
plt.show()





#  F1_score
from sklearn.metrics import f1_score
print("F1_score:", f1_score(Y_test_no_encoder, predicted_label, average='macro'))

## 可视化每个类别的ROC曲线
from sklearn.metrics import roc_curve,roc_curve, roc_auc_score

lable_names = ["K","N","P","health","white"]
colors = ["r","b","g","m","k",]
linestyles =["-", "--", "-.", ":", "-"]
pre_score = loaded_model.predict([X_test[:,0:9,:], X_test[:,9:18,:]])
print(pre_score)

print(Y_test)

# 画多分类的ROC曲线，就必须把预测的标签转换为[0，0，1，0，0]这种one_hot编码的形式
pre_test = np.zeros((200,5))
for i in range(len(Y_test)):
    index = np.argmax(pre_score[i,:])
    pre_test[i,index] = 1
print(pre_test)




fig  = plt.figure(figsize=(8,7))
for ii, color in zip(range(5), colors):
    ## 计算绘制ROC曲线的取值
    fpr_ii, tpr_ii, _ = roc_curve(Y_test[:, ii], pre_score[:, ii])
    auc = roc_auc_score(Y_test[:, ii], pre_score[:, ii])
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
    fpr_ii, tpr_ii, _ = roc_curve(Y_test[:, ii], pre_score[:, ii])
    ## 局部放大图
    inset_ax.plot(fpr_ii, tpr_ii,color = color,linewidth = 2,
                  linestyle = linestyles[ii])
    inset_ax.set_xlim([-0.01,0.06])
    inset_ax.set_ylim([0.8,1.01])
    inset_ax.grid()

plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240116CNN_concat\\CNN\\CNN ROC-AUC.svg')
plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240116CNN_concat\\CNN\\CNN ROC-AUC.png')
plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240116CNN_concat\\CNN\\CNN ROC-AUC.eps')
plt.show()


# 模型参数固定，精度随着训练样本增加的变化曲线
score = []
n = 160
t = int(800/n)
print(t)
for i in range(1, n+1):
    #estimator.fit(X_train, Y_train, validation_data=(X_test, Y_test))
    model.fit(x=[X_train[0:t*i-1,0:9,:], X_train[0:t*i-1,9:18,:]], y=Y_train[0:t*i-1])
    score_test = model.evaluate([X_test[:,0:9,:], X_test[:,9:18,:]], Y_test, verbose=0)[1]
    score.append(score_test)
print(score)
plt.plot(range(1,n+1), score)
plt.ylabel('Accuracy')
plt.xlabel('scale of train data')
plt.ylim(0,1)
plt.title('CNN Test Accuracy')
plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240116CNN_concat\\CNN\\160accuracy_data_scale.png', dpi = 600)    #保存图片一定要在plt.show()之前
plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240116CNN_concat\\CNN\\CNN accuracy_data_scale.eps', dpi = 600)    #保存图片一定要在plt.show()之前
plt.savefig('C:\\Users\\jiang\\Desktop\\all_data\\20240116CNN_concat\\CNN\\CNN accuracy_data_scale.svg')
plt.show()

score_to_excel = pd.DataFrame(data=score)
score_to_excel.to_excel(r'C:\\Users\\jiang\\Desktop\\all_data\\20240116CNN_concat\\CNN\\CNNscore.xlsx')






















