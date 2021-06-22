import pandas as pd
import numpy as np
import time
import os
import matplotlib as  mpl
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, \
    recall_score, f1_score

mpl.rcParams['font.sans-serif'] = ['simHei']
mpl.rcParams['axes.unicode_minus'] = False
pd.set_option("display.width",500)

def Cal_WD(time1,time2):
    '''计算到达晚点或者出发晚点'''
    time1 = pd.to_datetime(time1)
    time2 = pd.to_datetime(time2)
    return ((time1 - time2).total_seconds())/60
    ### example
    ##aa = Cal_WD('2016-11-03 14:06:00','2016-11-03 14:16:00')
    ###data3 ['到达晚点'] = list(map(Cal_WD, data3['实际到达时间'],data3['实际出发时间']))


def Modify_ColName(data):
    '''如果输入数据是直接从数据库导出，列名为拼音简写，此时修改为中文
    '''
    if 'DDCC' in  data.columns:
        data.rename(columns={'DDCC':'车次号'},inplace=True)
    if 'CCCF' in  data.columns:
        data.rename(columns={'CCCF':'出发车次'},inplace=True)
    if 'NODE' in  data.columns:
        data.rename(columns={'NODE':'车站'},inplace=True)
    if 'DDSJ' in  data.columns:
        data.rename(columns={'DDSJ':'实际到达时间'},inplace=True)
    if 'CFSJ' in  data.columns:
        data.rename(columns={'CFSJ':'实际出发时间'},inplace=True)
    if 'TDDDSJ' in  data.columns:
        data.rename(columns={'TDDDSJ':'图定到达时间'},inplace=True)
    if 'TDCFSJ' in  data.columns:
        data.rename(columns={'TDCFSJ':'图定出发时间'},inplace=True)
    if 'TRACK_NAME' in  data.columns:
        data.rename(columns={'TRACK_NAME':'股道号'},inplace=True)
    return data

def TurnTimetoStand(timestra):
    '''将时间标准化 转化结果如：2015/1/4 => 2015-01-04 00:00:00'''
    ta = pd.to_datetime(timestra)
    # ta = datetime.datetime.strptime(timestra, '%Y/%m/%d')
    # ta = str(ta)
    return ta
    ##example
    ##aa = '2015/3/24 10:00'
    ##aa1 = TurnTimetoStand(aa) ---- 2015-03-24 10:00:00

def Cal_DelayPeriod(timeStra,format = '数字'):  ####时间格式是斜杆的晚点对应的晚点时段
    '''计算晚点时段 '''
    ta = pd.to_datetime(timeStra)
    ta = time.strptime(str(ta), "%Y-%m-%d %H:%M:%S")
    H, M = ta.tm_hour , ta.tm_min
    date_ta = (int(H) * 60 + int(M)) / 1440

    if format == '数字':
        return date_ta
    ###example
    ###aa = Cal_DelayPeriod('2016-11-03 14:06:00') ----'0.5875'
    if format == '字符串':
        label  = pd.cut([date_ta],[0,0.25, 0.291666666666667, 0.333333333333333, 0.375, 0.416666666666667, 0.458333333333334, 0.5,
        0.541666666666667, 0.583333333333333, 0.625, 0.666666666666667, 0.708333333333333, 0.75,
        0.791666666666667, 0.833333333333333, 0.875, 0.916666666666666, 0.958333333333333, 1],
        right=True, labels=['0:00-6:00','6:00-7:00', '7:00-8:00','8:00-9:00', '9:00-10:00','10:00-11:00', '11:00-12:00', '12:00-13:00',
                            '13:00-14:00','14:00-15:00', '15:00-16:00', '16:00-17:00', '17:00-18:00', '18:00-19:00','19:00-20:00',
                            '20:00-21:00', '21:00-22:00', '22:00-23:00', '23:00-24:00'])
        return label.T[0]
def Cal_date(time1,format = '横线' ):
	'''计算日期'''
	time1 = pd.to_datetime(time1)
	if  format == '横线':
		return str(time1).split()[0]
	elif format == '斜线':
		ta = time.strptime(str(time1), "%Y-%m-%d %H:%M:%S")
		date_ta1 = str(ta.tm_year) + '/' + str(ta.tm_mon) + '/' + str(ta.tm_mday)
		return date_ta1
    ###example
    ###aa = Cal_date('2016-11-03 14:06:00','斜线') ----'2016/11/3'
    ####data['日期'] =  list(map(Cal_date,data['图定到达时间']))
    ###data['日期'] =  list(map(Cal_date,data['图定到达时间'],['横线']*len(data)))

def print_TTAT_precison(y, y_hat, algorithm='算法', dataset='测试集', Dayinzhibiao=True):
    # print("==" * 30)
    result = np.abs(y - y_hat)
    result_0 = result[result == 0]
    result_1 = result[result <= 1]
    result_2 = result[result <= 2]
    result_3 = result[result <= 3]
    result_4 = result[result <= 4]
    result_5 = result[result <= 5]
    result_10 = result[result <= 10]
    acc = 100 * len(result_0) / len(y)
    lessthan1 = 100 * len(result_1) / len(y_hat)
    lessthan2 = 100 * len(result_2) / len(y_hat)
    lessthan3 = 100 * len(result_3) / len(y_hat)
    lessthan4 = 100 * len(result_4) / len(y_hat)
    lessthan5 = 100 * len(result_5) / len(y_hat)
    lessthan10 = 100 * len(result_10) / len(y_hat)
    r2 = r2_score(y, y_hat)
    mae = mean_absolute_error(y, y_hat)
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    mse = mean_squared_error(y, y_hat)
    if Dayinzhibiao == True:
        print("{}{}R2：".format(algorithm, dataset), r2_score(y, y_hat))
        print("{}{}MAE：".format(algorithm, dataset), mean_absolute_error(y, y_hat))
        # print("{}{}平均绝对误差(MAPE)：".format(algorithm,dataset),np.mean(abs(y-y_hat)/y))
        print("{}{}准确率为：".format(algorithm, dataset), acc)
        print("{}{}1min误差：".format(algorithm, dataset), lessthan1)
        print("{}{}2min误差：".format(algorithm, dataset), lessthan2)
        print("{}{}3min误差：".format(algorithm, dataset), lessthan3)
        print("{}{}4min误差：".format(algorithm, dataset), lessthan4)
        print("{}{}5min误差：".format(algorithm, dataset), lessthan5)
        print("{}{}10min误差：".format(algorithm, dataset), lessthan10)
        print("{}{}的RMSE为：".format(algorithm, dataset), rmse)
        print("{}{}的MSE为：".format(algorithm, dataset), mse)
    if 0 in y or 0 in y_hat:
        pass
    else:
        MAPE = np.mean(abs(y - y_hat) / y)
        if Dayinzhibiao == True:
            print("{}{}平均绝对误差(MAPE)：".format(algorithm, dataset), np.mean(abs(y - y_hat) / y))



def print_NAT_precison(y,y_hat,algorithm ='RF',dataset = '训练集'):
    print("{}{}accuracy：".format(algorithm, dataset),accuracy_score(y,y_hat))
    print("{}{}precision_score：".format(algorithm, dataset),precision_score(y,y_hat,average=None))
    print("{}{}recall_score：".format(algorithm, dataset),recall_score(y,y_hat,average=None))
    print("{}{}f1_score：".format(algorithm, dataset),f1_score(y,y_hat,average=None))









