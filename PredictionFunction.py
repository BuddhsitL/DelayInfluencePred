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
from SomeFunction import Cal_WD,Cal_date,Modify_ColName,TurnTimetoStand,Cal_DelayPeriod
from TransitionData_Standard import transition_data


print('请输入待预测的列车所在车站')
Input_Station = input()

print('请输入待预测的列车车次号')
Input_TrainNo = input()


print('请输入待预测的实际到达时间')
Input_ActualArrive = input()
Input_ActualArrive = TurnTimetoStand(Input_ActualArrive)
Input_Date = Cal_date(str(Input_ActualArrive))

if os.path.exists('./PreProcessedData/%s车站时刻表数据.csv' % Input_Station):
	time_table = pd.read_csv('./PreProcessedData/%s车站时刻表数据.csv' % Input_Station)
	time_table_date = time_table.loc[time_table['日期']==Input_Date,:]
else:
	transition_data()
	time_table = pd.read_csv('./PreProcessedData/%s车站时刻表数据.csv' % Input_Station)
	time_table_date = time_table.loc[time_table['日期'] == Input_Date, :]
	# time_table['日期'] = time_table['日期'].map(Cal_date)



DDWD = Cal_WD(Input_ActualArrive, time_table_date.loc[time_table_date['车次']==Input_TrainNo,'图定到达时间'].max())
Whether_stop = time_table_date.loc[time_table_date['车次']==Input_TrainNo,'是否停站'].max()
Delay_period = Cal_DelayPeriod(Input_ActualArrive)
time_table_date.loc[:,'与实际到达时间差值'] = list(map(Cal_WD,time_table_date.loc[:,'图定到达时间'].values,[Input_ActualArrive]*time_table_date.shape[0]))
Minimum_difference = time_table_date.loc[time_table_date['与实际到达时间差值']>0,:].loc[:,'与实际到达时间差值'].min()
Next_trainNo = time_table_date.loc[time_table_date['与实际到达时间差值'] == Minimum_difference,'车次'].max()
Next_train_PlanArrive = time_table_date.loc[time_table_date['车次'] == Next_trainNo,'图定到达时间'].max()

Actual_interval = Cal_WD(Input_ActualArrive,Next_train_PlanArrive)
Whether_SameStationTrack = 0 if time_table_date.loc[time_table_date['车次'] == Next_trainNo,'股道号'].max() == time_table_date.loc[time_table_date['车次'] == Input_TrainNo,'股道号'].max() else 0

Five_minInterval_ideal_NAT = 0
Five_minInterval_ideal_TTAT = 0

from DelayDataCleaning import Cal_indicator


cum2,k = 0,0
Rest_SupTime = DDWD
sum2 = DDWD
time_table_date.loc[:,'原始索引'] = time_table_date.index
time_table_date.index = range(time_table_date.shape[0])
j = np.argwhere(np.array(time_table_date['车次'] == Input_TrainNo))[0][0]
Predict_train_index = time_table_date.loc[time_table_date['车次'] == Input_TrainNo].index.values[0]
while DDWD > cum2:
	cum2 = cum2 + time_table_date.loc[j + k, "以5min为间隔的冗余时间"]
	if Rest_SupTime > time_table_date.loc[j + k, "以5min为间隔的冗余时间"]:
		# time_table_date.loc[j + k, "以5min为间隔的冗余时间"] > 0:
		sum2 = sum2 + time_table_date.loc[j + k, "到达晚点"] - time_table_date.loc[j + k, "以5min为间隔的冗余时间"]
	k += 1
	Five_minInterval_ideal_NAT = k
	Five_minInterval_ideal_TTAT = sum2


NAT_Model = joblib.load('./modelsave/%s_影响列车数model.m'%Input_Station)

Model_NAT_input = pd.DataFrame({"到达晚点":DDWD,"实际间隔时间":Actual_interval,"是否停站":Whether_stop,"晚点时段":Delay_period,
                           "前方列车是否与后方列车共用一条到发线":Whether_SameStationTrack,
                            "以5min为间隔的理想恢复影响车数":Five_minInterval_ideal_NAT},index = [Predict_train_index])
NAT_predict = NAT_Model.predict(Model_NAT_input)

#######预测TTAT
TTAT_Model = joblib.load('./modelsave/%s_影响总时间model.m'%Input_Station)
Model_TTAT_input = pd.DataFrame({"到达晚点":DDWD,"实际间隔时间":Actual_interval,"是否停站":Whether_stop,"晚点时段":Delay_period,
                           "前方列车是否与后方列车共用一条到发线":Whether_SameStationTrack,
                            "以5min为间隔的理想恢复影响车数":Five_minInterval_ideal_NAT,
							"以5min为间隔的理想影响总时间":Five_minInterval_ideal_TTAT,
							"预测的影响列车数":NAT_predict},index = [Predict_train_index])

TTAT_predict = TTAT_Model.predict(Model_TTAT_input)
# print('列车{}的晚点将会造成{}列车的连带晚点,所有列车的晚点总时间为{}'.format(Input_TrainNo,NAT_predict,np.rint(TTAT_predict)))
Influenced_trainNo = time_table_date.loc[Predict_train_index:Predict_train_index + NAT_predict[0]-1,'车次'].values

print('列车{}的晚点将会造成{}列车的连带晚点,所有列车的晚点总时间为{}min'.format(Input_TrainNo,Influenced_trainNo,np.rint(TTAT_predict)))










