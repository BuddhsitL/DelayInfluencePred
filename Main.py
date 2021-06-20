import pandas as pd
import numpy as np
import time
from  sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostClassifier,AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score,mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import LabelBinarizer
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
import os
import matplotlib as  mpl
mpl.rcParams['font.sans-serif'] = ['simHei']
mpl.rcParams['axes.unicode_minus'] = False
pd.set_option("display.width",500)
import operator
import joblib
from SomeFunction import Cal_WD,Cal_date,Modify_ColName,TurnTimetoStand,Cal_DelayPeriod,print_NAT_precison



#%%
pd.set_option("display.width",300)
print('请输入车站名称：')
station_name = input()

if os.path.exists('./AfterFeatureExtraction/%s车站晚点横向传播特征数据提取.csv'%station_name):
    print(True)
    result1 = pd.read_csv('./AfterFeatureExtraction/%s车站晚点横向传播特征数据提取.csv'%station_name,encoding = 'UTF-8')

else:
    station_name1 = '%s车站数据.csv'%station_name
    data = pd.read_csv(station_name1,encoding="gbk")
    data.index = range(data.shape[0])
    print('原始数据是否是数据库直接导出：','请输入','  是  ','或者','  否  ')
    if_fromdatabase = input()
    data = data.drop_duplicates()

    if if_fromdatabase == '是':
        data1 = Modify_ColName(data)
    else:
        print('修改原始数据列名')
        data1 = data.rename(columns={'日期.y': '日期', '车站.y': '车站', '实际到达.y': '实际到达时间', '实际出发.y': '实际出发时间',
                                     '图定到达.y': '图定到达时间', '图定出发.y': '图定出发时间', '轨道.y': '股道号'}, inplace=False)

    ######
    print('计算晚点时段')
    data1['晚点时段'] = data1['实际到达时间'].map(Cal_DelayPeriod)

    ######
    print('将原始数据时间数据格式更为标准格式')
    for i in ['图定到达时间','图定出发时间','实际到达时间','实际出发时间']:
        data1[i] = data1[i].map(TurnTimetoStand)

    ######
    print('计算到达晚点')
    data1['到达晚点'] = [Cal_WD(data1.loc[i,'实际到达时间'],data1.loc[i,'图定到达时间'])for i in data1.index]

    ######
    print('计算是否停站')
    data1['是否停站'] = [0 if data1.loc[i,'图定到达时间'] == data1.loc[i,'图定出发时间'] else 1 for i in data1.index]

    ######计算间隔时间

    for i,(indicator,interval) in enumerate(zip(['实际到达时间','图定到达时间'],['实际间隔时间','图定间隔时间'])):
        if i == 0:
            data1.sort_values(by=indicator, ascending=True, inplace=True)
            data1.index = range(data1.shape[0])
            print('计算前方列车是否与后方列车共用一条到发线')
            if_same_track	= [1 if data1.loc[i+1,'股道号'] == data1.loc[i,'股道号'] else 0 for i in range(data1.shape[0] - 1)]
            if_same_track.append('last one interval')
            data1['前方列车是否与后方列车共用一条到发线']  = if_same_track
            print('计算 实际间隔时间，就是与下一列车的按实际到达时间排序后的间隔时间')

        else:
            print('计算 图定间隔时间，就是与下一列车的按图定到达时间排序后的间隔时间')

        data1.sort_values(by=indicator, ascending=True, inplace=True)
        data1.index = range(data1.shape[0])
        actual_interval = [Cal_WD(data1.loc[i + 1, indicator], data1.loc[i, indicator]) for i in
                           range(data1.shape[0] - 1)]
        actual_interval.append('last one interval')
        data1[interval] = actual_interval

    ######
    print('计算考虑停站的冗余时间')

    Con_stop_sup_time = [(data1.loc[i,'图定间隔时间'] - 3) if (data1.loc[i,'是否停站'] == 0 and data1.loc[i+1,'是否停站'] == 0) else (data1.loc[i,'图定间隔时间'] - 5) for i in range(data1.shape[0] - 1) ]
    Con_stop_sup_time.append('last indicator')
    data1['考虑停站的冗余时间'] = Con_stop_sup_time
    data1 = data1.loc[ data1['考虑停站的冗余时间']!='last indicator',:]
    data1.loc[data1['考虑停站的冗余时间'].astype(float) <0, '考虑停站的冗余时间'] = 0

    ######
    print('计算以5min为间隔的冗余时间')
    data1['以5min为间隔的冗余时间'] = data1['图定间隔时间'] - 5
    data1.loc[data1['以5min为间隔的冗余时间']<0, '以5min为间隔的冗余时间'] = 0

    ######
    # data1 = data1.iloc[1:,]
    data1.sort_values(by="实际到达时间", ascending=True, inplace=True)
    data_insert = pd.DataFrame(np.array([-100]*data1.shape[1]).reshape(1,data1.shape[1]),columns=data1.columns)
    data1 = pd.concat([data_insert,data1])
    data1.loc[:,"以5min为间隔的理想恢复影响车数"]=-10000
    data1.loc[:,"以5min为间隔的理想影响总时间"]=-10000
    data1.loc[:,"考虑停站的理想恢复影响车数"]=-10000
    data1.loc[:,"考虑停站的理想影响总时间"]=-10000


    ### 计算以5min为间隔的理想恢复影响车数，影响总时间，考虑停站的理想恢复影响车数，影响总时间
    col_name = data1.columns
    first_col = data1.iloc[0,:].values
    date=pd.unique(data1["日期"])
    data1 = data1.drop_duplicates()
    data1.index = range(data1.shape[0])
    result=np.array(col_name).reshape(1,-1)

    i=3
    while i<data1.shape[0]:
    #while i<2000:
        if data1.loc[i,"到达晚点"]>4 and data1.loc[i-3:i+3,"实际间隔时间"].min()>0:###不考虑越行
            cum1=0
            cum2=0
            sum2=0
            n=0
            k=0
            j=i
            sum1=data1.loc[j,"到达晚点"]
            sum2=data1.loc[j,"到达晚点"]
            while data1.loc[j,"到达晚点"]>cum1:
                cum1=cum1+data1.loc[j+n,"考虑停站的冗余时间"]
                if data1.loc[j+n,"到达晚点"]-data1.loc[j+n,"考虑停站的冗余时间"]>0:
                    sum1=sum1+data1.loc[j+n,"到达晚点"]-data1.loc[j+n,"考虑停站的冗余时间"]
                n+=1
                data1.loc[i,"考虑停站的理想恢复影响车数"]=n
                data1.loc[i,"考虑停站的理想影响总时间"]=sum1
            while data1.loc[j,"到达晚点"]>cum2:
                cum2=cum2+data1.loc[j+k,"以5min为间隔的冗余时间"]
                if data1.loc[j+k,"到达晚点"]-data1.loc[j+k,"以5min为间隔的冗余时间"]>0:
                    sum2=sum2+data1.loc[j+k,"到达晚点"]-data1.loc[j+k,"以5min为间隔的冗余时间"]
                k+=1
                data1.loc[i,"以5min为间隔的理想恢复影响车数"]=k
                data1.loc[i,"以5min为间隔的理想影响总时间"]=sum2
            result=np.vstack((result,np.array(data1.iloc[i,:])))
            m=1
            while data1.loc[i+m,"到达晚点"]>0 and Cal_WD(data1.loc[i+m,"图定到达时间"],data1.loc[i+m-1,"实际到达时间"])<5 :
                result=np.vstack((result,np.array(data1.iloc[i+m,:])))
                m=m+1
            result=np.vstack((result,first_col))
            i=i+m
        else:
            i=i+1
        # print(i)


    # print(result)
    result=pd.DataFrame(result,columns=col_name)
    result.iloc[0,] = np.array([-100]*data1.shape[1]).reshape(1,data1.shape[1])



    result.loc[result['车次']== -100, "日期"] = np.arange(1,result.loc[result['车次']== -100,:].shape[0]+1)

    for i in ["影响列车数","影响总时间","列车编号","晚点持续时长","平均间隔时间"]:
        result[i] = -1000

    result.loc[:,"序号"] =range(result.shape[0])
    X = result[result.loc[:, "车次"] == -100].shape[0]
    result1 = np.array(result.columns).reshape(1, -1)

    for i in np.arange(1, X):
        upper = result.loc[result ["日期"] == i, "序号"].max()  ###前行列车
        lower = result.loc[result ["日期"] == i + 1, "序号"].max()  ####后行列车
        data3 = result.iloc[upper + 1:lower, :]
        if data3.loc[:, "实际间隔时间"].min() > 0:  ##要使用按实际到达时间排序后相减的，因为不越行的话就是图定的间隔
            data3.loc[:, "影响列车数"] = data3.shape[0]
            data3.loc[:, "影响总时间"] = np.sum(data3.loc[:, "到达晚点"])
            data3.loc[:, "列车编号"] = np.arange(1, data3.shape[0] + 1)
            data3.loc[:, "晚点持续时长"] = Cal_WD(result.loc[lower - 1, "实际到达时间"], result.loc[upper + 1, "实际到达时间"]) + data3.loc[lower - 1, "到达晚点"]
            data3.loc[:, "平均间隔时间"] = np.mean(data3.loc[:, "图定间隔时间"])
            data3.index = data3["列车编号"]
            data3 = np.array(data3)
            result1 = np.vstack((result1, data3))
        print(i)
    print(result1)
    result1 = pd.DataFrame(result1)
    result1.to_csv('./AfterFeatureExtraction/%s车站晚点横向传播特征数据提取.csv'%station_name,encoding='utf_8_sig',header=0)

#%%

if os.path.exists('./AfterFeatureExtraction/%s车站晚点横向传播特征数据提取.csv'%station_name):
    print(True)
    result1 = pd.read_csv('./AfterFeatureExtraction/%s车站晚点横向传播特征数据提取.csv'%station_name,encoding = 'UTF-8')
result=result1.loc[result1["列车编号"] == 1,:]

result=result[["到达晚点","实际间隔时间","是否停站","前方列车是否与后方列车共用一条到发线","晚点时段","考虑停站的理想恢复影响车数"
          ,"考虑停站的理想影响总时间","以5min为间隔的理想影响总时间","以5min为间隔的理想恢复影响车数","影响列车数","影响总时间"]]

result=result.loc[result["影响列车数"]<15,:]
result["影响列车数1"]=result["影响列车数"]
result.loc[result["影响列车数"]>6,"影响列车数1"]=6

x=result[["到达晚点","实际间隔时间","是否停站","晚点时段","前方列车是否与后方列车共用一条到发线","以5min为间隔的理想恢复影响车数","影响总时间","以5min为间隔的理想影响总时间"]]
y=result["影响列车数1"]

x_train1,x_test1,y_train,y_test = train_test_split(x, y, train_size=0.7,test_size=0.3, random_state=3)
x_train=x_train1[["到达晚点","实际间隔时间","是否停站","晚点时段","前方列车是否与后方列车共用一条到发线","以5min为间隔的理想恢复影响车数"]]
x_test=x_test1[["到达晚点","实际间隔时间","是否停站","晚点时段","前方列车是否与后方列车共用一条到发线","以5min为间隔的理想恢复影响车数"]]


# print("="*30,"RF对NAT建模","="*30)

model=RandomForestClassifier(random_state=4,oob_score=False,n_estimators=500,criterion='entropy',max_depth=5,min_samples_leaf= 1, min_samples_split= 6 )
model.fit(x_train,y_train)
rf_train_hat=model.predict(x_train)
rf_test_hat=model.predict(x_test)



if os.path.exists('./modelsave/%s_影响列车数model.m'%station_name):  # 如果训练好了 就加载一下不用反复训练
    model = joblib.load('./modelsave/%s_影响列车数model.m'%station_name)##调用模型
    # rf_train_hat = model.predict(x_train)
else:
    model = RandomForestClassifier(random_state=4, oob_score=False, n_estimators=500, criterion='entropy', max_depth=3,
                                   min_samples_leaf=1, min_samples_split=6)
    model.fit(x_train, y_train)
    # rf_train_hat = model.predict(x_train)
    joblib.dump(model, './modelsave/%s_影响列车数model.m'%station_name)

'''RF预测NAT精度展示'''
##打印随机森林训练集精度
# rf_test_hat = bst.predict(y_test)
# print_NAT_precison(y_train,rf_train_hat,"XGBOOST",'训练集')
# print_NAT_precison(y_test,rf_test_hat,"XGBOOST",'测试集')





# print("~"*30,"影响总时间预测","~"*30)

x_td_train=x_train1.copy()
x_td_train["预测的影响列车数"]=rf_train_hat
x_td_test=x_test1.copy()
x_td_test["预测的影响列车数"]=rf_test_hat

data_td=pd.concat([x_td_train,x_td_test],axis=0)
data_td=data_td[data_td["影响总时间"]<60]

#5min为间隔
x_td=data_td[["到达晚点","实际间隔时间","是否停站","晚点时段","前方列车是否与后方列车共用一条到发线","以5min为间隔的理想恢复影响车数","以5min为间隔的理想影响总时间","预测的影响列车数"]]
y_td=data_td["影响总时间"]

x_train_td,x_test_td,y_train_td,y_test_td=train_test_split(x_td,y_td,train_size=0.7,test_size=0.3,random_state=4)




from sklearn import svm
from SomeFunction import print_TTAT_precison
##打印影响总时间训练集精度

# print("="*30,"SVR对TTAT建模","="*30)
##SVR预测影响总时间

if os.path.exists('./modelsave/%s_影响总时间model.m'%station_name):  # 如果训练好了 就加载一下不用反复训练
    model_svr = joblib.load('./modelsave/%s_影响总时间model.m'%station_name)##调用模型
else:
    model_svr = svm.SVR(kernel="rbf", C=100, gamma=0.001, epsilon=0.001)
    model_svr.fit(x_train_td, y_train_td)
    joblib.dump(model_svr, './modelsave/%s_影响总时间model.m'%station_name)


'''SVR预测精度展示'''
# td_svm_y_train_hat = np.rint(model_svr.predict(x_train_td))
# td_svm_y_test_hat = np.rint(model_svr.predict(x_test_td))
# print_TTAT_precison(y_train_td,td_svm_y_train_hat,"SVR",'训练集')
# print("=="*30)
# print_TTAT_precison(y_test_td,td_svm_y_test_hat,"SVR",'测试集')















