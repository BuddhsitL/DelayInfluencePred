#-*-coding:utf-8-*-
import pandas as pd
import numpy as np
from  sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostClassifier,AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score,mean_squared_error,r2_score,mean_absolute_error,roc_auc_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn import neighbors
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder,LabelBinarizer
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,auc
import  matplotlib as  mpl
import operator

mpl.rcParams['font.sans-serif'] = ['simHei']
mpl.rcParams['axes.unicode_minus'] = False
pd.set_option("display.width",500)

##打印影响列车数训练集精度
def print_NAT_train_precison(y,y_hat,algorithm):    
    print("%s训练集测试精度："%algorithm,accuracy_score(y,y_hat))
    print("%s训练集测试precision_score："%algorithm,precision_score(y,y_hat,average=None))
    print("%s训练集测试recall_score："%algorithm,recall_score(y,y_hat,average=None))
    print("%s训练集测试f1_score："%algorithm,f1_score(y,y_hat,average=None))  
    
def print_NAT_test_precison(y,y_hat,algorithm):
    print("%s测试集精度："%algorithm,accuracy_score(y,y_hat))
    print("%s测试集precision_score："%algorithm,precision_score(y,y_hat,average=None))
    print("%s测试集recall_score："%algorithm,recall_score(y,y_hat,average=None))
    print("%s测试集f1_score："%algorithm,f1_score(y,y_hat,average=None))
    
def print_NAT_2018test_precison(y,y_hat,algorithm):  
    print("%s2018年训练集精度："%algorithm,accuracy_score(y,y_hat))
    print("%s2018年训练集precision_score："%algorithm,precision_score(y,y_hat,average=None))
    print("%s2018年训练集recall_score："%algorithm,recall_score(y,y_hat,average=None))
    print("%s2018年训练集f1_score："%algorithm,f1_score(y,y_hat,average=None))

##打印影响总时间训练集精度
def print_TTAT_train_precison(y,y_hat,algorithm):    
    print("=="*30)
    result=np.abs(y-y_hat)
    result_1=result[result<=1]
    result_2=result[result<=2]
    result_3=result[result<=3]
    result_4=result[result<=4]
    result_5=result[result<=5]
    result_10=result[result<=10]
    print("%s训练集测试R2："%algorithm,r2_score(y,y_hat))
    print("%s训练集测试MAE："%algorithm,mean_absolute_error(y,y_hat))
    print("%s训练集平均绝对误差(MAPE)："%algorithm,np.mean(abs(y-y_hat)/y))
    print("%s训练集1min误差："%algorithm,100*len(result_1)/len(y_hat))
    print("%s训练集2min误差："%algorithm,100*len(result_2)/len(y_hat))
    print("%s训练集3min误差："%algorithm,100*len(result_3)/len(y_hat))
    print("%s训练集4min误差："%algorithm,100*len(result_4)/len(y_hat))
    print("%s训练集5min误差："%algorithm,100*len(result_5)/len(y_hat))
    print("%s训练集10min误差："%algorithm,100*len(result_10)/len(y_hat))
def print_TTAT_test_precison(y,y_hat,algorithm):  
    print("=="*30)
    result=np.abs(y-y_hat)
    result_1=result[result<=1]
    result_2=result[result<=2]
    result_3=result[result<=3]
    result_4=result[result<=4]
    result_5=result[result<=5]
    result_10=result[result<=10]
    print("%s测试集R2："%algorithm,r2_score(y,y_hat))
    print("%s测试集测试MAE："%algorithm,mean_absolute_error(y,y_hat))
    print("%s测试集平均绝对误差(MAPE)："%algorithm,np.mean(abs(y-y_hat)/y))
    print("%s测试集1min误差："%algorithm,100*len(result_1)/len(y_hat))
    print("%s测试集2min误差："%algorithm,100*len(result_2)/len(y_hat))
    print("%s测试集3min误差："%algorithm,100*len(result_3)/len(y_hat))
    print("%s测试集4min误差："%algorithm,100*len(result_4)/len(y_hat))
    print("%s测试集5min误差："%algorithm,100*len(result_5)/len(y_hat))
    print("%s测试集10min误差："%algorithm,100*len(result_10)/len(y_hat))
def print_TTAT_2018test_precison(y,y_hat,algorithm):  
    print("=="*30)
    result=np.abs(y-y_hat)
    result_1=result[result<=1]
    result_2=result[result<=2]
    result_3=result[result<=3]
    result_4=result[result<=4]
    result_5=result[result<=5]
    result_10=result[result<=10]
    print("%s2018年测试集R2："%algorithm,r2_score(y,y_hat))
    print("%s2018年测试集MAE："%algorithm,mean_absolute_error(y,y_hat))
    print("%s2018年测试集平均绝对误差(MAPE)："%algorithm,np.mean(abs(y-y_hat)/y))
    print("%s2018年测试集1min误差："%algorithm,100*len(result_1)/len(y_hat))
    print("%s2018年测试集2min误差："%algorithm,100*len(result_2)/len(y_hat))
    print("%s2018年测试集3min误差："%algorithm,100*len(result_3)/len(y_hat))
    print("%s2018年测试集4min误差："%algorithm,100*len(result_4)/len(y_hat))
    print("%s2018年测试集5min误差："%algorithm,100*len(result_5)/len(y_hat))
    print("%s2018年测试集10min误差："%algorithm,100*len(result_10)/len(y_hat))



data=open("广州北车站晚点横向提取(新).csv","r",encoding="UTF-8")
data=pd.read_csv(data)
data=data[data.loc[:,"列车编号"]==1]

data=data[["到达晚点","首列间隔时间","是否停站","前方列车是否与后方列车共用一条到发线","晚点时段","考虑停站的理想恢复影响车数"
          ,"考虑停站的理想影响总时间","以5min为间隔的理想影响总时间","以5min为间隔的理想恢复影响车数","影响列车数","影响总时间"]]
data=data[data.loc[:,"影响列车数"]<15]
data["影响列车数1"]=data["影响列车数"]
data.loc[data["影响列车数"]>6,"影响列车数1"]=6

#data["roc_auc"]=data["影响列车数1"]
data=data.sort_values(by=["影响列车数1"])
lb =LabelBinarizer()
y_auc=lb.fit_transform(data["影响列车数1"])
#y_auc1=lb.inverse_transform(y_auc)


#data["label"]=pd.cut(data["晚点时段"],[0,0.291666666666667,0.333333333333333,0.375,0.416666666666667,0.458333333333334,0.5,0.541666666666667,0.583333333333333,0.625,0.666666666666667,0.708333333333333,0.75,0.791666666666667,0.833333333333333,0.875,0.916666666666666,0.958333333333333,1],
#                     right=True, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18])   ##将晚点时段划分区间
#data["label"]=pd.to_numeric(data["label"])
#data["影响列车数1"]=pd.to_numeric(data["影响列车数1"])
#Le = LabelEncoder()
#data["label1"]=Le.fit_transform(data["label"])
## # print(data["label"])
#label=np.array(data["label"]).reshape(-1,1)
#data["label1"]=OneHotEncoder(sparse = False).fit_transform(label)
## print(label)
# print(data)
#x=data[["到达晚点","首列间隔时间","是否停站","晚点时段","理想恢复影响车数","影响总时间"]]



###以5min为间隔的理性恢复车数的X
x=data[["到达晚点","首列间隔时间","是否停站","晚点时段","前方列车是否与后方列车共用一条到发线","以5min为间隔的理想恢复影响车数","影响总时间","以5min为间隔的理想影响总时间"]]

##考虑停站的理性恢复车数的X
#x=data[["到达晚点","首列间隔时间","是否停站","晚点时段","前方列车是否与后方列车共用一条到发线","考虑停站的理想恢复影响车数","影响总时间","考虑停站的理想影响总时间"]]
# print(x.head())
y=data["影响列车数1"]
#y=lb.fit_transform(y)

# y=np.array(y).reshape(1,-1)
x_train1,x_test1,y_train,y_test = train_test_split(x, y, train_size=0.7,test_size=0.3, random_state=3)


####以5min为间隔的理性恢复车数的X
x_train=x_train1[["到达晚点","首列间隔时间","是否停站","晚点时段","前方列车是否与后方列车共用一条到发线","以5min为间隔的理想恢复影响车数"]]
x_test=x_test1[["到达晚点","首列间隔时间","是否停站","晚点时段","前方列车是否与后方列车共用一条到发线","以5min为间隔的理想恢复影响车数"]]

y_train_auc=lb.fit_transform(y_train)
y_test_auc=lb.fit_transform(y_test)
#####考虑停站的理想恢复车数的X
#x_train=x_train1[["到达晚点","首列间隔时间","是否停站","晚点时段","前方列车是否与后方列车共用一条到发线","考虑停站的理想恢复影响车数"]]
#x_test=x_test1[["到达晚点","首列间隔时间","是否停站","晚点时段","前方列车是否与后方列车共用一条到发线","考虑停站的理想恢复影响车数"]]

# x_train1,x_test1,y_train1,y_test1 = train_test_split(x1, y1, train_size=0.9,test_size=0.1, random_state=3)


#######2018年车站数据验证
data1=open("2018广州北车站晚点横向提取(新).csv","r",encoding="UTF-8")
data1=pd.read_csv(data1)
data1=data1[data1.loc[:,"列车编号"]==1]
data1=data1[["到达晚点","首列间隔时间","是否停站","前方列车是否与后方列车共用一条到发线","晚点时段","考虑停站的理想恢复影响车数"
          ,"考虑停站的理想影响总时间","以5min为间隔的理想影响总时间","以5min为间隔的理想恢复影响车数","影响列车数","影响总时间"]]
data1=data1[data1.loc[:,"影响列车数"]<15]
data1["影响列车数1"]=data1["影响列车数"]
data1.loc[data1["影响列车数"]>6,"影响列车数1"]=6


data1["label"]=pd.cut(data1["晚点时段"],[0,0.291666666666667,0.333333333333333,0.375,0.416666666666667,0.458333333333334,0.5,0.541666666666667,0.583333333333333,0.625,0.666666666666667,0.708333333333333,0.75,0.791666666666667,0.833333333333333,0.875,0.916666666666666,0.958333333333333,1],
                     right=True, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18])

#x11=data1[["到达晚点","首列间隔时间","是否停站","晚点时段","理想恢复影响车数","影响总时间"]]###原来的X11

x11=data1[["到达晚点","首列间隔时间","是否停站","晚点时段","前方列车是否与后方列车共用一条到发线","以5min为间隔的理想恢复影响车数","以5min为间隔的理想影响总时间","影响总时间"]]

#x1=x11[["到达晚点","首列间隔时间","是否停站","晚点时段","理想恢复影响车数"]]
x1=x11[["到达晚点","首列间隔时间","是否停站","晚点时段","前方列车是否与后方列车共用一条到发线","以5min为间隔的理想恢复影响车数"]]
# x1=np.array(x1)
y1=data1["影响列车数1"]
# y1=np.array(y1)
y_auc2018=lb.fit_transform(y1)

# ssx=StandardScaler()
# x=ssx.fit_transform(x)
# x1=ssx.fit_transform(x1)

#####================================================================================================================#####
##热力图绘制
#
import seaborn as sns
sns.set(font_scale=1.5)
data["label"]=pd.cut(data["晚点时段"],[0,0.291666666666667,0.333333333333333,0.375,0.416666666666667,0.458333333333334,0.541666666666667,0.583333333333333,0.625,0.666666666666667,0.708333333333333,0.75,0.791666666666667,0.833333333333333,0.875,0.916666666666666,0.958333333333333,1],
                     right=True, labels=["7:00-8:00","8:00-9:00" , "9:00-10:00","10:00-11:00" , "11:00-12:00","12:00-13:00" ,"13:00-14:00","14:00-15:00" , "15:00-16:00","16:00-17:00" , "17:00-18:00","18:00-19:00" , "19:00-20:00","20:00-21:00" , "21:00-22:00","22:00-23:00" ,"23:00-24:00"])


# data["label"]=pd.cut(data["晚点时段"],[0,0.458333333333334,0.625,0.791666666666667,1],
#                      right=True, labels=["7:00-11:00","11:00-15:00" , "15:00-19:00","19:00-00:00"])
data["label1"]=pd.cut(data["到达晚点"],[4,10,15,20,25,30,100],
                     right=True, labels=["5--10","10--15" , "15--20","20--25" , "25--30","30--100"],)

data_heatmap=data[["label1","label","影响列车数"]]
data_heatmap=data_heatmap.sort_values(by=['label1'])
pt = data_heatmap.pivot_table(index='label1', columns='label', values='影响列车数',aggfunc=np.mean)
pt=pt.fillna(0)
data_heatmap.to_csv("热力图.csv")
print(pt)
print(type(pt))
pt.to_csv("3D.csv")
# f, (ax1,ax2) = plt.subplots(figsize = (100,20),nrows=2)
plt.subplots(figsize = (10,10))
plt.grid(b=True, ls=':', color='k')
sns.heatmap(pt, linewidths = 0,  vmin=-1,cmap="hot_r")
# rainbow为 matplotlib 的colormap名称
# plt.title('影响列车数热力图',fontsize=16)
# plt.xlabel('晚点时段',fontsize=16)
# plt.ylabel('到达晚点',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Heatmap of NAT at GZN station(Average)',fontsize=16)
plt.xlabel('Period of primary delay',fontsize=16)
plt.ylabel('Primary delay duration(min)',fontsize=16)
plt.show()


#####
'''NAT总数热力图显示'''
data["label1"]=pd.cut(data["到达晚点"],[4,10,15,20,25,30,100],
                     right=True, labels=["5--10","10--15" , "15--20","20--25" , "25--30","30--100"],)

data_heatmap=data[["label1","label","影响列车数"]]
pt = data_heatmap.pivot_table(index='label1', columns='label', values='影响列车数',aggfunc=np.sum)
pt=pt.fillna(0)
data_heatmap.to_csv("广州北NAT总数热力图.csv")
pt.to_csv("广州北NAT总数3D.csv")





#####================================================================================================================#####



print("="*30,"RF对NAT建模","="*30)
#随机森林算法
##广州北
model=RandomForestClassifier(random_state=4,oob_score=False,n_estimators=500,criterion='entropy',max_depth=5,min_samples_leaf= 1, min_samples_split= 6 )

# param={"max_depth":np.arange(1,10),"min_samples_leaf":np.arange(1,10),"min_samples_split":np.arange(5,20)}##"max_feature":np.arange(1,6),
# model=GridSearchCV(model,param)
model.fit(x_train,y_train)

# print(model.best_params_)
rf_train_hat=model.predict(x_train)
rf_test_hat=model.predict(x_test)

rf_result_train=rf_train_hat==y_train
rf_result_test=rf_test_hat==y_test


'''RF预测NAT精度展示'''
##打印随机森林训练集精度
print_NAT_train_precison(y_train,rf_train_hat,"rf")
print("=="*30)
##打印随机森林测试集精度
print_NAT_test_precison(y_test,rf_test_hat,"rf")

# print(model.feature_importances_)
# important_features = pd.DataFrame(data={'features': x_train.columns, 'importance': model.feature_importances_})
# important_features.sort_values(by='importance', axis=0, ascending=False, inplace=True)
# important_features['cum_importance'] = important_features['importance'].cumsum()
# print('特征重要度：\n', important_features


print("="*30,"2018年RF对NAT验证","="*30)
##2017年预测
rf_train_hat2018=model.predict(x1)
print_NAT_2018test_precison(y1,rf_train_hat2018,"RF")


y_score1=model.fit(x_train,y_train).predict_proba(x_test)
fpr1, tpr1, thresholds = roc_curve(y_test_auc.ravel(),y_score1.ravel())
auc1 = auc(fpr1, tpr1)
print(auc1)


#####================================================================================================================#####


print("="*30,"XGBOOST对NAT建模","="*30)
#xgboost算法
###影响列车数预测

data_train=xgb.DMatrix(x_train,y_train)
data_test=xgb.DMatrix(x_test)
watch_list=[(data_test,"eval"),(data_train,"train")]


##广州北
param={"booster":"gbtree","max_depth":4,"eta":0.1,"silent":1,'min_child_weight': 2,"objective":"multi:softmax","num_class":8}
# model=GridSearchCV(estimator = XGBClassifier( learning_rate =0.1)
#                    ,param_grid={ "max_depth":np.arange(2,10),"min_child_weight":np.arange(1,10)})
model.fit(x_train,y_train)
# print(model.best_params_)


bst=xgb.train(param,data_train,num_boost_round=10)
xgb_train_hat=bst.predict(xgb.DMatrix(x_train))
xgb_test_hat=bst.predict(data_test)


###XGBOOST调参
###max_depth调参
#for i in np.arange(1,15):
#    print(i)
#    param={"booster":"gbtree","max_depth":i,"eta":0.1,"silent":1,'min_child_weight':1,"objective":"multi:softmax","num_class":8}
#    # model=GridSearchCV(estimator = XGBClassifier( learning_rate =0.1)
#    #                    ,param_grid={ "max_depth":np.arange(2,10),"min_child_weight":np.arange(1,10)})
#    model.fit(x_train,y_train)
#    # print(model.best_params_)
#    bst=xgb.train(param,data_train,num_boost_round=10)
#    xgb_train_hat=bst.predict(xgb.DMatrix(x_train))
#    xgb_test_hat=bst.predict(data_test)
#    data_2017_test=xgb.DMatrix(x1)
#    xgb_test_predict2017=bst.predict(xgb.DMatrix(x1))
#    print_NAT_train_precison(y_test,xgb_test_hat,"XGBOOST")
#    print_NAT_2018test_precison(y1,xgb_test_predict2017,"XGBOOST")
#    print("=="*30)


'''XGBoost预测NAT精度展示'''
##打印随机森林训练集精度
print_NAT_train_precison(y_train,xgb_train_hat,"XGBOOST")
print("=="*30)
##打印随机森林测试集精度
print_NAT_train_precison(y_test,xgb_test_hat,"XGBOOST")


importance = bst.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))
print(importance)
importance = bst.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df.loc[df["feature"]=="是否停站","feature"]="B"
df.loc[df["feature"]=="前方列车是否与后方列车共用一条到发线","feature"]="A"
df.loc[df["feature"]=="首列间隔时间","feature"]="I"
df.loc[df["feature"]=="到达晚点","feature"]="D"
df.loc[df["feature"]=="以5min为间隔的理想恢复影响车数","feature"]="N"
df.loc[df["feature"]=="晚点时段","feature"]="T"
df['fscore'] = df['fscore'] / df['fscore'].sum()
plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', fontsize=18, figsize=(20, 10),legend=False)#legend=False,
#plt.title('XGBOOST特征重要度', fontsize=18)
#plt.ylabel("特  征", fontsize=18)
#plt.xlabel('重要度', fontsize=18)
plt.grid(b=True, ls=':', color='k')
plt.title('Feature Importance of XGBOOST Algorithm at GZN Station', fontsize=18)
plt.ylabel("Feature", fontsize=18)
plt.xlabel('Improtance', fontsize=18)
plot_importance(bst)
plt.show()
#plt.savefig('GZN_feature_importance_xgb.png')

print("="*30,"XGBOOST2018年验证结果","="*30)
##2018年验证数据
data_2017_test=xgb.DMatrix(x1)
xgb_test_predict2017=bst.predict(xgb.DMatrix(x1))
print_NAT_2018test_precison(y1,xgb_test_predict2017,"XGBOOST")

#####================================================================================================================#####
#fpr=dict()
#tpr=dict()
#roc_auc=dict()
#
#for i in np.arange(n_class):
#    fpr[i],tpr[i],threshold = roc_curve(y_test[:,i], y_score[:,i])
#    roc_auc[i] = auc(fpr[i], tpr[i])

'''绘制XGBOOST的auc曲线'''

y_train1=y_train-1
y_test1=y_test-1
data_train1=xgb.DMatrix(x_train,y_train1)
data_test1=xgb.DMatrix(x_test)
watch_list=[(data_test1,"eval"),(data_train1,"train")]
param1={"booster":"gbtree","max_depth":4,"eta":0.1,"silent":1,'min_child_weight': 2,"objective":"multi:softprob","num_class":6}##

bst1=xgb.train(param1,data_train1,num_boost_round=10)
xgb_train_hat_auc=bst1.predict(xgb.DMatrix(x_train))
xgb_test_hat_auc=bst1.predict(data_test1)

y_score2=bst1.predict(data_test1)
fpr2, tpr2, thresholds1 = roc_curve(y_test_auc.ravel(),y_score2.ravel())
auc2 = auc(fpr2, tpr2)
print(auc2)

#####================================================================================================================#####





print("="*30,"Adbsoot对NAT建模","="*30)
#Adaboost
base_estimator = DecisionTreeClassifier(criterion='gini', max_depth=6,max_features=4, min_samples_split=4)
clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=10, learning_rate=0.1)
clf.fit(x_train, y_train.ravel())
Adb_y_train_hat=clf.predict(x_train)
Adb_y_test_hat=clf.predict(x_test)

##
print_NAT_train_precison(y_train,Adb_y_train_hat,"Adaboost")
##
print_NAT_test_precison(y_test,Adb_y_test_hat,"Adaboost")
#
#####================================================================================================================#####

##SVM做NAT
print("="*30,"SVM对NAT建模","="*30)
model_svm=svm.SVC(kernel="rbf",C=1,gamma=0.001,decision_function_shape="ovr",probability=True)
# param1={"C":np.logspace(-3,3,10),"gamma":np.logspace(-3,3,10)}
# model_svm=GridSearchCV(model_svm,param1)
model_svm.fit(x_train,y_train)
# print(model_svm.best_params_)
svm_y_train_hat=model_svm.predict(x_train)
svm_y_test_hat=model_svm.predict(x_test)


'''SVM预测NAT精度展示'''
##SVM训练集精度
print_NAT_train_precison(y_train,svm_y_train_hat,"SVM")
print("=="*30)
##SVN测试集精度
print_NAT_test_precison(y_test,svm_y_test_hat,"SVM")

y_score3=model_svm.fit(x_train,y_train).predict_proba(x_test)
fpr3, tpr3, thresholds = roc_curve(y_test_auc.ravel(),y_score3.ravel())
auc3 = auc(fpr3, tpr3)
#print(roc_auc_score(y_test_auc,y_score3))
print( auc(fpr3, tpr3))





#####================================================================================================================#####


print("="*30,"LogisticRegression对NAT建模","="*30)
##logictic模型
LR_model = LogisticRegression(C=100)
# param2={"C":np.logspace(-3, 4, 6)}
# LR_model=GridSearchCV(LR_model,param2)
LR_model.fit(x_train, y_train)
# print(LR_model.best_params_)
LR_y_train_hat = LR_model.predict(x_train)
LR_y_test_hat = LR_model.predict(x_test)


'''LogisticRegression预测NAT精度展示'''
##LogisticRegression训练集精度
print_NAT_train_precison(y_train,LR_y_train_hat,"LogisticRegression")
##LogisticRegression测试集精度
print_NAT_test_precison(y_test,LR_y_test_hat,"LogisticRegression")
y_score4=model_svm.fit(x_train,y_train).predict_proba(x_test)


'''LogisticRegression的ROC与auc展示'''
fpr4, tpr4, thresholds = roc_curve(y_test_auc.ravel(),y_score4.ravel())
auc4 = auc(fpr4, tpr4)
print(roc_auc_score(y_test_auc,y_score4))
print(auc(fpr4, tpr4))



#####================================================================================================================#####


print("="*30,"KNN对NAT建模","="*30)
##KNN分类器
KNN_model = neighbors.KNeighborsClassifier(n_neighbors=3)
KNN_model.fit(x_train, y_train)
KNN_y_train_hat = KNN_model.predict(x_train)
KNN_y_test_hat= KNN_model.predict(x_test) 

'''KNN预测NAT精度展示'''
##knn训练集精度
print_NAT_train_precison(y_train,KNN_y_train_hat,"knn")
print("=="*30)
##knn测试集精度
print_NAT_test_precison(y_test,KNN_y_test_hat,"knn")


'''KNN的ROC与auc展示'''
y_score5=KNN_model.fit(x_train,y_train).predict_proba(x_test)
fpr5, tpr5, thresholds = roc_curve(y_test_auc.ravel(),y_score5.ravel())
auc5 = auc(fpr5, tpr5)
print(roc_auc_score(y_test_auc,y_score5))
print( auc(fpr5, tpr5))

#####================================================================================================================#####
'''输出各个模型的NAT结果'''
def output_ResultNAT():
    result_NAT=[[accuracy_score(y_test,rf_test_hat),auc(fpr1,tpr1),accuracy_score(y1,rf_train_hat2018)],##RF
                 [accuracy_score(y_test,xgb_test_hat),auc(fpr2,tpr2),accuracy_score(y1,xgb_test_predict2017)],##XGBOOST
                 [accuracy_score(y_test,svm_y_test_hat),auc(fpr3,tpr3),"NAN"],##SVM
                 [accuracy_score(y_test,LR_y_test_hat,),auc(fpr4,tpr4),"NAN"],##LR
                 [accuracy_score(y_test,KNN_y_test_hat,),auc(fpr5,tpr5),"NAN"]]##KNN
    result_NAT=pd.DataFrame(result_NAT,index=["RF","XGBOOST","SVM","LR","KNN"],columns=["accuracy","auc","2018accuracy"])
    return result_NAT
result_NAT=output_ResultNAT()

#####================================================================================================================#####

'''绘制ROC曲线与AUC值图'''
def print_roc_auc():
    print(roc_auc_score(y_test_auc,y_score1))
    fpr1, tpr1, thresholds1 = roc_curve(y_test_auc.ravel(),y_score1.ravel())
    fpr2, tpr2, thresholds1 = roc_curve(y_test_auc.ravel(),y_score2.ravel())
    fpr3, tpr3, thresholds1 = roc_curve(y_test_auc.ravel(),y_score3.ravel())
    fpr4, tpr4, thresholds1 = roc_curve(y_test_auc.ravel(),y_score4.ravel())
    fpr5, tpr5, thresholds1 = roc_curve(y_test_auc.ravel(),y_score5.ravel())
    auc1 = auc(fpr1, tpr1)
    auc2 = auc(fpr2, tpr2)
    auc3 = auc(fpr3, tpr3)
    auc4 = auc(fpr4, tpr4)
    auc5 = auc(fpr5, tpr5)
    #绘图
    plt.subplots(figsize=(10, 10))
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    #FPR就是横坐标,TPR就是纵坐标
    plt.plot(fpr1, tpr1, c = 'r', lw = 2, alpha = 0.7, ls = ':' , label = u'RF_AUC=%.4f' % auc1)
    plt.plot(fpr2, tpr2, c = 'b', lw = 2, alpha = 0.7,ls = '-', label = u'XGBOOST_AUC=%.4f' % auc2)
    plt.plot(fpr3, tpr3, c = 'y', lw = 2, alpha = 0.7, ls = '--', label = u'SVM_AUC=%.4f' % auc3)
    plt.plot(fpr4, tpr4, c = 'darkorange', lw = 2,ls = '-.', alpha = 0.7, label = u'LR_AUC=%.4f' % auc4)
    plt.plot(fpr5, tpr5, c = 'cornflowerblue', lw = 2, ls = '--',alpha = 0.7,  label = u'KNN_AUC=%.4f' % auc5)
    plt.plot((0, 1), (0, 1), c = '#808080', lw = 2, ls = '--', alpha = 0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1),fontsize=16)
    plt.yticks(np.arange(0, 1.1, 0.1),fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=16)
    # plt.title('各种机器学习对NAT的ROC和AUC', fontsize=16)
    plt.title('ROC curve and AUC value of NAT at GZN station  ', fontsize=16)
    plt.show()
print_roc_auc()

#####================================================================================================================#####



print("~"*30,"影响总时间预测","~"*30)
##影响总时间预测
###建模数据
# x_td_train=pd.concat([x_train,xgb_train_predict],axis=1)
# y_td_train=pd.concat([y_train,xgb_test_predict],axis=1)
# data_td=pd.concat([x_td_train,y_td_train],axis=1)
x_td_train=x_train1
x_td_train=x_td_train.copy()
x_td_train["预测的影响列车数"]=xgb_train_hat
x_td_test=x_test1
x_td_test=x_td_test.copy()
x_td_test["预测的影响列车数"]=xgb_test_hat
data_td=pd.concat([x_td_train,x_td_test],axis=0)
data_td=data_td[data_td["影响总时间"]<60]
# data_td=data_td[data_td["预测的影响列车数"]==3]
# data_td=data_td[data_td["预测的影响列车数"]<=5]
#5min为间隔
x_td=data_td[["到达晚点","首列间隔时间","是否停站","晚点时段","前方列车是否与后方列车共用一条到发线","以5min为间隔的理想恢复影响车数","以5min为间隔的理想影响总时间","预测的影响列车数"]]
y_td=data_td["影响总时间"]

x_train_td,x_test_td,y_train_td,y_test_td=train_test_split(x_td,y_td,train_size=0.7,test_size=0.3,random_state=4)

##2018年数据组合
x11=data1[["到达晚点","首列间隔时间","是否停站","晚点时段","前方列车是否与后方列车共用一条到发线","以5min为间隔的理想恢复影响车数","以5min为间隔的理想影响总时间","影响总时间"]]
x_td_train2017=x11
x_td_train2017=x_td_train2017.copy()

x_td_train2017["预测的影响列车数"]=xgb_test_predict2017
x_td_train2017=x_td_train2017[x_td_train2017["影响总时间"]<60]

y11=x_td_train2017["影响总时间"]


#x=data[["到达晚点","首列间隔时间","是否停站","晚点时段","前方列车是否与后方列车共用一条到发线","考虑停站的理想恢复影响车数","影响总时间","考虑停站的理想影响总时间"]]
##5min为间隔
x_td_train2017=x_td_train2017[["到达晚点","首列间隔时间","是否停站","晚点时段","前方列车是否与后方列车共用一条到发线","以5min为间隔的理想恢复影响车数","以5min为间隔的理想影响总时间","预测的影响列车数"]]

#####================================================================================================================#####

##晚点总时间热力图绘制
import seaborn as sns
sns.set(font_scale=1.5)
data_td["TIME"]=pd.cut(data_td["晚点时段"],[0,0.291666666666667,0.333333333333333,0.375,0.416666666666667,0.458333333333334,0.541666666666667,0.583333333333333,0.625,0.666666666666667,0.708333333333333,0.75,0.791666666666667,0.833333333333333,0.875,0.916666666666666,0.958333333333333,1],
                      right=True, labels=["7:00-8:00","8:00-9:00" , "9:00-10:00","10:00-11:00" , "11:00-12:00","12:00-13:00" ,"13:00-14:00","14:00-15:00" , "15:00-16:00","16:00-17:00" , "17:00-18:00","18:00-19:00" , "19:00-20:00","20:00-21:00" , "21:00-22:00","22:00-23:00" ,"23:00-24:00"])


# data["label"]=pd.cut(data["晚点时段"],[0,0.458333333333334,0.625,0.791666666666667,1],
#                      right=True, labels=["7:00-11:00","11:00-15:00" , "15:00-19:00","19:00-00:00"])
# data_td["TD"]=pd.cut(data_td["影响总时间"],[4,15,25,35,45,55,60],
 #                      right=True, labels=["5—15","15—25" , "25—35","35—45" , "45—55","55—60"],)
data_td["DELAY"]=pd.cut(data_td["到达晚点"],[4,10,15,20,25,30,100],
                    right=True, labels=["5--10","10--15" , "15--20","20--25" , "25--30","30--100"],)
data_heatmap=data_td[["DELAY","TIME","影响总时间"]]
data_heatmap=data_heatmap.sort_values(by=['DELAY'])
pt = data_heatmap.pivot_table(index='DELAY', columns='TIME', values='影响总时间',aggfunc=np.mean)
pt=pt.fillna(0)
data_heatmap.to_csv("晚点总时间热力图.csv")
print(pt)
print(type(pt))
pt.to_csv("晚点总时间3D.csv")
# f, (ax1,ax2) = plt.subplots(figsize = (100,20),nrows=2)
plt.subplots(figsize = (10,10))
plt.grid(b=True, ls=':', color='k')
sns.heatmap(pt, linewidths = 0,  vmax=60, vmin=-1,cmap="hot_r")
# rainbow为 matplotlib 的colormap名称
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Heatmap of TTAT at GZN station(Average)',fontsize=16)
plt.xlabel('Period of primary delay',fontsize=16)
plt.ylabel('Primary delay duration(min)',fontsize=16)
#plt.title('影响总时间热力图',fontsize=16)
#plt.xlabel('晚点时段',fontsize=16)
#plt.ylabel('到达晚点',fontsize=16)
plt.show()



'''TTAT影响总时间热力图'''
pt = data_heatmap.pivot_table(index='DELAY', columns='TIME', values='影响总时间',aggfunc=np.sum)
pt=pt.fillna(0)
data_heatmap.to_csv("广州北TTAT热力图.csv")
print(pt)
print(type(pt))
pt.to_csv("广州北TTAT3D.csv")
#####================================================================================================================#####





print("="*30,"RF对TTAT建模","="*30)
##随机森林预测
model_rf_td=RandomForestRegressor(random_state=4,oob_score=False,n_estimators=500,criterion='mae',max_depth=5, min_samples_leaf= 1, min_samples_split= 5)
# param={"max_depth":np.arange(1,10),"min_samples_leaf":np.arange(1,10),"min_samples_split":np.arange(5,20)}##"max_feature":np.arange(1,6),
# model_rf_td=GridSearchCV(model_rf_td,param)
model_rf_td.fit(x_train_td,y_train_td)
# print(model_rf_td.best_params_)

td_rf_train_hat=model_rf_td.predict(x_train_td)
td_rf_test_hat=model_rf_td.predict(x_test_td)
td_rf_test_hat=np.rint(td_rf_test_hat)

##随机森林测试集与训练集精度

'''RF预测TTAT精度展示'''
print_TTAT_train_precison(y_train_td,td_rf_train_hat,"RF") 
print("=="*30)
print_TTAT_test_precison(y_test_td,td_rf_test_hat,"RF") 


print("="*30,"RF2018年对TTAT检验","="*30)
rf_td_y_test_hat2017= model_rf_td.predict(x_td_train2017)
td_rf_result_train2017=np.abs(y11-rf_td_y_test_hat2017)
td_rf_result_train2017=td_rf_result_train2017[td_rf_result_train2017<=10]
print_TTAT_2018test_precison(y11,rf_td_y_test_hat2017,"RF")






#####================================================================================================================#####

print("="*30,"XGBOOST对TTAT建模","="*30)
##xgboost算法
data_train_td=xgb.DMatrix(x_train_td,y_train_td)
data_test_td=xgb.DMatrix(x_test_td)
#衡阳东
xgb_td_model = xgb.XGBRegressor(max_depth=10, learning_rate=0.1, n_estimators=100,silent=True, objective='reg:gamma')#min_child_weight=1
#广州北
# xgb_td_model = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=160, min_child_weight=5, silent=True, objective='reg:gamma')
xgb_td_model.fit(x_train_td, y_train_td)


# 对测试集进行预测
xgb_td_y_train_hat=xgb_td_model.predict(x_train_td)
xgb_td_y_test_hat= xgb_td_model.predict(x_test_td)
xgb_td_y_train_hat=np.rint(xgb_td_y_train_hat)

#####================================================================================================================#####

'''xgboost预测精度展示'''
print_TTAT_train_precison(y_train_td,xgb_td_y_train_hat,"XGBOOST") 
print("=="*30)
print_TTAT_test_precison(y_test_td,xgb_td_y_test_hat,"XGBOOST") 

print("="*30,"XGBOOST2018年对TTAT检验","="*30)


xgb_td_y_test_hat2017= xgb_td_model.predict(x_td_train2017)
print_TTAT_2018test_precison(y11,xgb_td_y_test_hat2017,"XGBOOST")
 






#####================================================================================================================#####

print("="*30,"SVR对TTAT建模","="*30)
##SVR预测影响总时间
##广州北
# model_svr=svm.SVR(kernel="rbf",C=100,gamma=0.001,epsilon=0.001)
###衡阳东
model_svr=svm.SVR(kernel="rbf",C=10000,gamma=0.0001,epsilon=0.001)
# param1={"C":np.logspace(-3,3,10),"gamma":np.logspace(-3,3,10)}
# model_svr=GridSearchCV(model_svr,param1)
model_svr.fit(x_train_td,y_train_td)
# print(model_svr.best_params_)
td_svm_y_train_hat=model_svr.predict(x_train_td)
td_svm_y_test_hat=model_svr.predict(x_test_td)
td_svm_y_test_hat=np.rint(td_svm_y_test_hat)

#SVR调参
#for i in [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]:
#    print(i)
#    model_svr=svm.SVR(kernel="rbf",C=10000,gamma=0.0001,epsilon=0.001)
#    model_svr.fit(x_train_td,y_train_td)
#    svr_td_y_test_hat2017= model_svr.predict(x_td_train2017)
#    td_svr_result_train2017=np.abs(y11-svr_td_y_test_hat2017)
#    td_svr_result_train2017=td_svr_result_train2017[td_svr_result_train2017<=10]
#    print_TTAT_2018test_precison(y11,svr_td_y_test_hat2017,"SVR")
#    print("=="*30)

'''SVR预测精度展示'''
print_TTAT_train_precison(y_train_td,td_svm_y_train_hat,"SVR") 
print("=="*30)
print_TTAT_test_precison(y_test_td,td_svm_y_test_hat,"SVR") 

print("="*30,"SVR对2018年TTAT检验","="*30)

#SVR做2017年预测
svr_td_y_test_hat2017= model_svr.predict(x_td_train2017)
td_svr_result_train2017=np.abs(y11-svr_td_y_test_hat2017)
td_svr_result_train2017=td_svr_result_train2017[td_svr_result_train2017<=10]
print_TTAT_2018test_precison(y11,svr_td_y_test_hat2017,"SVR")



#####================================================================================================================#####


print("="*30,"Ridge对TTAT建模","="*30)
##岭回归回归
model = Ridge(alpha=100)
alpha_can = np.logspace(-3, 2, 10)
# np.set_printoptions(suppress=True)
# print('alpha_can = ', alpha_can)
Ridge_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
Ridge_model.fit(x_train_td,y_train_td)
# print('超参数：\n', Ridge_model.best_params_)
# order = y_test.argsort(axis=0)
# y_test = y_test.values[order]
# x_test = x_test.values[order, :]
Ridge_y_hat_test = Ridge_model.predict(x_test_td)
Ridge_y_hat_train=Ridge_model.predict(x_train_td)
Ridge_y_hat_test=np.rint(Ridge_y_hat_test)


'''ridge预测精度展示'''
print_TTAT_train_precison(y_train_td,Ridge_y_hat_train,"Ridge") 
print("=="*30)
print_TTAT_test_precison(y_test_td,Ridge_y_hat_test,"Ridge") 

#####================================================================================================================#####

print("="*30,"Lasso对TTAT建模","="*30)
###Lasso回归
lasso_model = Lasso(alpha=2.154434690031882)
alpha_can = np.logspace(-3, 3, 10)
# np.set_printoptions(suppress=True)
# print('alpha_can = ', alpha_can)
# lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
lasso_model.fit(x_train_td,y_train_td)
# print('超参数：\n', lasso_model.best_params_)
# order = y_test.argsort(axis=0)
# y_test = y_test.values[order]
# x_test = x_test.values[order, :]
lasso_y_hat_test = lasso_model.predict(x_test_td)
lasso_y_hat_train=lasso_model.predict(x_train_td)
lasso_y_hat_test=np.rint(lasso_y_hat_test)


'''lasso预测精度展示'''
print_TTAT_train_precison(y_train_td,lasso_y_hat_train,"lasso") 
print("=="*30)
print_TTAT_test_precison(y_test_td,lasso_y_hat_test,"lasso") 


def output_ResulTTAT():
    '''顺序依次是R2,MAE,MAPE,LESSTHAN1,2,3,4,5'''  
    ###RF
    r2_RF= r2_score(y_test_td,td_rf_test_hat)##R2
    MAE_RF=mean_absolute_error(td_rf_test_hat,y_test_td)##MAE
    MAPE_RF=np.mean(abs(y_test_td-td_rf_test_hat)/y_test_td)##MAPE
    result_RF=np.abs(y_test_td-td_rf_test_hat)
    result_RF1= 100*len(result_RF[result_RF<=1])/len(y_test_td)##LESSTHAN1
    result_RF2= 100*len(result_RF[result_RF<=2])/len(y_test_td)##LESSTHAN2
    result_RF3= 100*len(result_RF[result_RF<=3])/len(y_test_td)##LESSTHAN3
    result_RF4= 100*len(result_RF[result_RF<=4])/len(y_test_td)##LESSTHAN4
    result_RF5= 100*len(result_RF[result_RF<=5])/len(y_test_td)##LESSTHAN5
       ###  XGBOOST
    r2_XGBOOST= r2_score(y_test_td,xgb_td_y_test_hat)##R2
    MAE_XGBOOST=mean_absolute_error(xgb_td_y_test_hat,y_test_td)##MAE
    MAPE_XGBOOST=np.mean(abs(y_test_td-xgb_td_y_test_hat)/y_test_td)##MAPE
    result_XGBOOST=np.abs(y_test_td-xgb_td_y_test_hat)
    result_XGBOOST1= 100*len(result_XGBOOST[result_XGBOOST<=1])/len(y_test_td)##LESSTHAN1
    result_XGBOOST2= 100*len(result_XGBOOST[result_XGBOOST<=2])/len(y_test_td)##LESSTHAN2
    result_XGBOOST3= 100*len(result_XGBOOST[result_XGBOOST<=3])/len(y_test_td)##LESSTHAN3
    result_XGBOOST4= 100*len(result_XGBOOST[result_XGBOOST<=4])/len(y_test_td)##LESSTHAN4
    result_XGBOOST5= 100*len(result_XGBOOST[result_XGBOOST<=5])/len(y_test_td)##LESSTHAN5
    ###  SVM
    r2_SVM= r2_score(y_test_td,td_svm_y_test_hat)##R2
    MAE_SVM=mean_absolute_error(td_svm_y_test_hat,y_test_td)##MAE
    MAPE_SVM=np.mean(abs(y_test_td-td_svm_y_test_hat)/y_test_td)##MAPE
    result_SVM=np.abs(y_test_td-td_svm_y_test_hat)
    result_SVM1= 100*len(result_SVM[result_SVM<=1])/len(y_test_td)##LESSTHAN1
    result_SVM2= 100*len(result_SVM[result_SVM<=2])/len(y_test_td)##LESSTHAN2
    result_SVM3= 100*len(result_SVM[result_SVM<=3])/len(y_test_td)##LESSTHAN3
    result_SVM4= 100*len(result_SVM[result_SVM<=4])/len(y_test_td)##LESSTHAN4
    result_SVM5= 100*len(result_SVM[result_SVM<=5])/len(y_test_td)##LESSTHAN5

    ##LASSO
    r2_LASSO= r2_score(y_test_td,lasso_y_hat_test)##R2
    MAE_LASSO=mean_absolute_error(lasso_y_hat_test,y_test_td)##MAE
    MAPE_LASSO=np.mean(abs(y_test_td-lasso_y_hat_test)/y_test_td)##MAPE
    result_LASSO=np.abs(y_test_td-lasso_y_hat_test)
    result_LASSO1= 100*len(result_LASSO[result_LASSO<=1])/len(y_test_td)##LESSTHAN1
    result_LASSO2= 100*len(result_LASSO[result_LASSO<=2])/len(y_test_td)##LESSTHAN2
    result_LASSO3= 100*len(result_LASSO[result_LASSO<=3])/len(y_test_td)##LESSTHAN3
    result_LASSO4= 100*len(result_LASSO[result_LASSO<=4])/len(y_test_td)##LESSTHAN4
    result_LASSO5= 100*len(result_LASSO[result_LASSO<=5])/len(y_test_td)##LESSTHAN5
       
    ##RIDGE
    r2_RIDGE= r2_score(y_test_td,Ridge_y_hat_test)##R2
    MAE_RIDGE=mean_absolute_error(Ridge_y_hat_test,y_test_td)##MAE
    MAPE_RIDGE=np.mean(abs(y_test_td-Ridge_y_hat_test)/y_test_td)##MAPE
    result_RIDGE=np.abs(y_test_td-Ridge_y_hat_test)
    result_RIDGE1= 100*len(result_RIDGE[result_RIDGE<=1])/len(y_test_td)##LESSTHAN1
    result_RIDGE2= 100*len(result_RIDGE[result_RIDGE<=2])/len(y_test_td)##LESSTHAN2
    result_RIDGE3= 100*len(result_RIDGE[result_RIDGE<=3])/len(y_test_td)##LESSTHAN3
    result_RIDGE4= 100*len(result_RIDGE[result_RIDGE<=4])/len(y_test_td)##LESSTHAN4
    result_RIDGE5= 100*len(result_RIDGE[result_RIDGE<=5])/len(y_test_td)##LESSTHAN5
    
    #####svr2018    
    
    r2_2018SVM= r2_score(y11,svr_td_y_test_hat2017)##R2
    MAE_2018SVM=mean_absolute_error(y11,svr_td_y_test_hat2017)##MAE
    MAPE_2018SVM=np.mean(abs(y11-svr_td_y_test_hat2017)/y11)##MAPE
    result_2018SVM=np.abs(y11-svr_td_y_test_hat2017)
    result_2018SVM1= 100*len(result_2018SVM[result_2018SVM<=1])/len(y11)##LESSTHAN1
    result_2018SVM2= 100*len(result_2018SVM[result_2018SVM<=2])/len(y11)##LESSTHAN2
    result_2018SVM3= 100*len(result_2018SVM[result_2018SVM<=3])/len(y11)##LESSTHAN3
    result_2018SVM4= 100*len(result_2018SVM[result_2018SVM<=4])/len(y11)##LESSTHAN4
    result_2018SVM5= 100*len(result_2018SVM[result_2018SVM<=5])/len(y11)##LESSTHAN5
    
    result_TTAT=[[r2_RF,MAE_RF,MAPE_RF,result_RF1,result_RF2,result_RF3,result_RF4,result_RF5],##RF
                 [r2_XGBOOST,MAE_XGBOOST,MAPE_XGBOOST,result_XGBOOST1,result_XGBOOST2,result_XGBOOST3,result_XGBOOST4,result_XGBOOST5],##XGBOOST
                 [r2_SVM,MAE_SVM,MAPE_SVM,result_SVM1,result_SVM2,result_SVM3,result_SVM4,result_SVM5],##SVM
                 [r2_LASSO,MAE_LASSO,MAPE_LASSO,result_LASSO1,result_LASSO2,result_LASSO3,result_LASSO4,result_LASSO5],##LR
                 [r2_RIDGE,MAE_RIDGE,MAPE_RIDGE,result_RIDGE1,result_RIDGE2,result_RIDGE3,result_RIDGE4,result_RIDGE5],
                 [r2_2018SVM,MAE_2018SVM,MAPE_2018SVM,result_2018SVM1,result_2018SVM2,result_2018SVM3,result_2018SVM4,result_2018SVM5]]##KNN
    result_TTAT=pd.DataFrame(result_TTAT,index=["RF","XGBOOST","SVR","LASSO","RIDGE","2018SVR"],columns=["R2","MAE","MAPE","LESSTHAN1","LESSTHAN2","LESSTHAN3","LESSTHAN4","LESSTHAN5"])
    return result_TTAT

result_TTAT=output_ResulTTAT()

result_TTAT.to_csv("D:/workspace/2019TRB_RESULT/广州北TTAT结果.csv",encoding="utf_8_sig")
result_NAT.to_csv("D:/workspace/2019TRB_RESULT/广州北NAT结果.csv",encoding="utf_8_sig")

print("="*30,"end","="*30)
    