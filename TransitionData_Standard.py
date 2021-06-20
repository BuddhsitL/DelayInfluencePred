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
from SomeFunction import Cal_WD,Cal_DelayPeriod,Modify_ColName,TurnTimetoStand


def transition_data():
	print('请输入车站名称：')
	station_name = input()

	station_name1 = '%s车站数据.csv' % station_name
	data = pd.read_csv(station_name1, encoding="gbk")
	data.index = range(data.shape[0])
	print('原始数据是否是数据库直接导出：', '请输入', '  是  ', '或者', '  否  ')
	if_fromdatabase = input()
	data = data.drop_duplicates()

	######
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
	for i in ['图定到达时间', '图定出发时间', '实际到达时间', '实际出发时间']:
		data1[i] = data1[i].map(TurnTimetoStand)

	######
	print('计算到达晚点')
	data1['到达晚点'] = [Cal_WD(data1.loc[i, '实际到达时间'], data1.loc[i, '图定到达时间']) for i in data1.index]

	######
	print('计算是否停站')
	data1['是否停站'] = [0 if data1.loc[i, '图定到达时间'] == data1.loc[i, '图定出发时间'] else 1 for i in data1.index]

	######计算间隔时间

	for i, (indicator, interval) in enumerate(zip(['实际到达时间', '图定到达时间'], ['实际间隔时间', '图定间隔时间'])):
		if i == 0:
			data1.sort_values(by=indicator, ascending=True, inplace=True)
			data1.index = range(data1.shape[0])
			print('计算前方列车是否与后方列车共用一条到发线')
			if_same_track = [1 if data1.loc[i + 1, '股道号'] == data1.loc[i, '股道号'] else 0 for i in range(data1.shape[0] - 1)]
			if_same_track.append('last one interval')
			data1['前方列车是否与后方列车共用一条到发线'] = if_same_track
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

	Con_stop_sup_time = [(data1.loc[i, '图定间隔时间'] - 3) if (data1.loc[i, '是否停站'] == 0 and data1.loc[i + 1, '是否停站'] == 0) else (
					data1.loc[i, '图定间隔时间'] - 5) for i in range(data1.shape[0] - 1)]
	Con_stop_sup_time.append('last indicator')
	data1['考虑停站的冗余时间'] = Con_stop_sup_time
	data1 = data1.loc[data1['考虑停站的冗余时间'] != 'last indicator', :]
	data1.loc[data1['考虑停站的冗余时间']< 0, '考虑停站的冗余时间'] = 0

	######
	print('计算以5min为间隔的冗余时间')
	data1['以5min为间隔的冗余时间'] = data1['图定间隔时间'] - 5
	data1.loc[data1['以5min为间隔的冗余时间'] < 0, '以5min为间隔的冗余时间'] = 0
	data1.to_csv('./PreProcessedData/%s车站时刻表数据.csv'%station_name,encoding='utf_8_sig')
	return data1

# data1 = transition_data()