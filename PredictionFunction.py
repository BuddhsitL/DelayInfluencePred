import pandas as pd
import numpy as np
import time
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

print('请输入初始晚点列车的晚点时间')
DDWD = int(input())

print('请输入日期')
Input_Date = input()
Input_Date = Cal_date(str(Input_Date))



if os.path.exists('./PreProcessedData/%s车站时刻表数据.csv' % Input_Station):
	time_table = pd.read_csv('./PreProcessedData/%s车站时刻表数据.csv' % Input_Station)
	time_table_date = time_table.loc[time_table['日期']==Input_Date,:].copy()
else:
	transition_data()
	time_table = pd.read_csv('./PreProcessedData/%s车站时刻表数据.csv' % Input_Station)
	time_table_date = time_table.loc[time_table['日期'] == Input_Date, :].copy()
	# time_table['日期'] = time_table['日期'].map(Cal_date)

# print('请输入待预测的实际到达时间')
# Input_ActualArrive = input()
Input_ActualArrive = pd.to_datetime(time_table_date.loc[time_table_date['车次']==Input_TrainNo,'图定到达时间'].max()) + \
                     pd.Timedelta(minutes = int(DDWD))
# Input_Date = Cal_date(str(Input_ActualArrive))

# DDWD = Cal_WD(Input_ActualArrive, time_table_date.loc[time_table_date['车次']==Input_TrainNo,'图定到达时间'].max())
Whether_stop = time_table_date.loc[time_table_date['车次']==Input_TrainNo,'是否停站'].max()
Delay_period = Cal_DelayPeriod(Input_ActualArrive)
time_table_date.loc[:,'与实际到达时间差值'] = list(map(Cal_WD,list(time_table_date.loc[:,'图定到达时间'].values),[Input_ActualArrive]*time_table_date.shape[0]))
Minimum_difference = time_table_date.loc[time_table_date['与实际到达时间差值']>0,:].copy().loc[:,'与实际到达时间差值'].min()
# Minimum_difference = Minimum_difference.loc[:,'与实际到达时间差值'].min()
Next_trainNo = time_table_date.loc[time_table_date['与实际到达时间差值'] == Minimum_difference,'车次'].max()
Next_train_PlanArrive = time_table_date.loc[time_table_date['车次'] == Next_trainNo,'图定到达时间'].max()


Sec_interval,Thi_interval,Fou_interval,Fiv_interval,Six_interval = time_table_date.loc[time_table_date['车次'] == Input_TrainNo,['与第2列间隔时间', '与第3列间隔时间', '与第4列间隔时间', '与第5列间隔时间', '与第6列间隔时间']].values[0]
Actual_interval = Cal_WD(Input_ActualArrive,Next_train_PlanArrive)
Whether_SameStationTrack = 0 if time_table_date.loc[time_table_date['车次'] == Next_trainNo,'股道号'].max() == time_table_date.loc[time_table_date['车次'] == Input_TrainNo,'股道号'].max() else 0

Five_minInterval_ideal_NAT = 0
Five_minInterval_ideal_TTAT = 0

from DelayDataCleaning import Cal_indicator


cum2,k = 0,0
Rest_SupTime = int(DDWD)
sum2 = int(DDWD)
time_table_date.loc[:,'原始索引'] = time_table_date.index.values
time_table_date.index = range(time_table_date.shape[0])
j = np.argwhere(np.array(time_table_date['车次'] == Input_TrainNo))[0][0]
Predict_train_index = time_table_date.loc[time_table_date['车次'] == Input_TrainNo].index.values[0]
while DDWD > cum2:
	cum2 = cum2 + time_table_date.loc[j + k, "以5min为间隔的冗余时间"]
	# print(2)
	if Rest_SupTime > time_table_date.loc[j + k, "以5min为间隔的冗余时间"]:
		# time_table_date.loc[j + k, "以5min为间隔的冗余时间"] > 0:
		sum2 = sum2 + time_table_date.loc[j + k, "到达晚点"] - time_table_date.loc[j + k, "以5min为间隔的冗余时间"]
		# print(3)
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
# print('列车{}的晚点将会造成{}列车的连带晚点,所有列车的晚点总时间为{}'.format(Input_TrainNo,NAT_predict,np.round(TTAT_predict)))
Influenced_trainNo = time_table_date.loc[Predict_train_index:Predict_train_index + NAT_predict[0]-1,'车次'].values


'''# # # # # # 预测第二列晚点列车晚点时间# # ## # # # # '''
which_train = 2
Sec_Model = joblib.load('./modelsave/%s第%s列车的晚点时间预测model.m' % (Input_Station, which_train))

Model_Sec_input = pd.DataFrame({"到达晚点": DDWD, "与第2列间隔时间": Sec_interval, "是否停站": Whether_stop, "晚点时段": Delay_period,
                                "前方列车是否与后方列车共用一条到发线": Whether_SameStationTrack,
                                "以5min为间隔的理想恢复影响车数": Five_minInterval_ideal_NAT}, index=[Predict_train_index])
Sec_prediction = np.round(Sec_Model.predict(Model_Sec_input))

'''# # # # # # 预测第三列晚点列车晚点时间# # ## # # # # '''
which_train = 3
Thi_Model = joblib.load('./modelsave/%s第%s列车的晚点时间预测model.m' % (Input_Station, which_train))

Model_Thi_input = pd.DataFrame({"到达晚点": DDWD, "与第2列间隔时间": Thi_interval, "是否停站": Whether_stop, "晚点时段": Delay_period,
                                "前方列车是否与后方列车共用一条到发线": Whether_SameStationTrack,
                                "以5min为间隔的理想恢复影响车数": Five_minInterval_ideal_NAT}, index=[Predict_train_index])

Thi_prediction = np.round(Thi_Model.predict(Model_Thi_input))

'''# # # # # # 预测第四列晚点列车晚点时间# # ## # # # # '''
which_train = 4
Fou_Model = joblib.load('./modelsave/%s第%s列车的晚点时间预测model.m' % (Input_Station, which_train))

Model_Fou_input = pd.DataFrame({"到达晚点": DDWD, "与第2列间隔时间": Fou_interval, "是否停站": Whether_stop, "晚点时段": Delay_period,
                                "前方列车是否与后方列车共用一条到发线": Whether_SameStationTrack,
                                "以5min为间隔的理想恢复影响车数": Five_minInterval_ideal_NAT}, index=[Predict_train_index])
Fou_prediction = np.round(Fou_Model.predict(Model_Fou_input))

'''# # # # # # 预测第五列晚点列车晚点时间# # ## # # # # '''
which_train = 5
Fiv_Model = joblib.load('./modelsave/%s第%s列车的晚点时间预测model.m' % (Input_Station, which_train))

Model_Fiv_input = pd.DataFrame({"到达晚点": DDWD, "与第5列间隔时间": Fiv_interval, "是否停站": Whether_stop, "晚点时段": Delay_period,
                                "前方列车是否与后方列车共用一条到发线": Whether_SameStationTrack,
                                "以5min为间隔的理想恢复影响车数": Five_minInterval_ideal_NAT}, index=[Predict_train_index])
Fiv_prediction = np.round(Fiv_Model.predict(Model_Fiv_input))

'''# # # # # # 预测第六列晚点列车晚点时间# # ## # # # # '''

which_train = 6
Six_Model = joblib.load('./modelsave/%s第%s列车的晚点时间预测model.m' % (Input_Station, which_train))

Model_Six_input = pd.DataFrame({"到达晚点": DDWD, "与第5列间隔时间": Six_interval, "是否停站": Whether_stop, "晚点时段": Delay_period,
                                "前方列车是否与后方列车共用一条到发线": Whether_SameStationTrack,
                                "以5min为间隔的理想恢复影响车数": Five_minInterval_ideal_NAT}, index=[Predict_train_index])
Six_prediction = np.round(Six_Model.predict(Model_Six_input))



if NAT_predict == 1 :
	print('列车{}的晚点将会造成{}列车的晚点,列车{}的晚点时间分别为{}min'.format(Input_TrainNo, Influenced_trainNo,Influenced_trainNo ,DDWD))

if NAT_predict == 2:
	print('列车{}的晚点将会造成{}列车的晚点,列车{}的晚点时间分别为{},{}min'.format(Input_TrainNo,
	                                                      Influenced_trainNo, Influenced_trainNo, DDWD,Sec_prediction))
if NAT_predict == 3:
	print('列车{}的晚点将会造成{}列车的晚点,列车{}的晚点时间分别为{},{},{}min'.format(Input_TrainNo, Influenced_trainNo, Influenced_trainNo, DDWD,
	                                                      Sec_prediction,Thi_prediction))
if NAT_predict == 4:
	print('列车{}的晚点将会造成{}列车的晚点,列车{}的晚点时间分别为{},{},{},{}min'.format(Input_TrainNo, Influenced_trainNo, Influenced_trainNo, DDWD,
	                                                      Sec_prediction,Thi_prediction,Fou_prediction))
if NAT_predict == 5:
	print('列车{}的晚点将会造成{}列车的晚点,列车{}的晚点时间分别为{},{},{},{},{}min'.format(Input_TrainNo, Influenced_trainNo, Influenced_trainNo, DDWD,
	                                                        Sec_prediction,Thi_prediction,Fou_prediction,Fiv_prediction))

if NAT_predict == 6:
	print('列车{}的晚点将会造成{}列车的晚点,列车{}的晚点时间分别为{},{},{},{},{},{}min'.format(Input_TrainNo, Influenced_trainNo, Influenced_trainNo, DDWD,
	                                                        Sec_prediction, Thi_prediction, Fou_prediction,Fiv_prediction,Six_prediction))


