# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:05:19 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:39:09 2018

@author: Administrator
"""

#-*-coding:utf-8-*-
import numpy as np
import pandas as pd
import datetime
import time
from DelayDataCleaning import DataCleaning
def getTimeDiff(timeStra,timeStrb):
    if timeStra<=timeStrb:
        return 0
    ta = time.strptime(timeStra, "%Y/%m/%d %H:%M")
    tb = time.strptime(timeStrb, "%Y/%m/%d %H:%M")
    y,m,d,H,M = ta[0:5]
    dataTimea=datetime.datetime(y,m,d,H,M)
    y,m,d,H,M = tb[0:5]
    dataTimeb=datetime.datetime(y,m,d,H,M)
    secondsDiff=(dataTimea-dataTimeb).seconds
    #两者相加得转换成分钟的时间差
    minutesDiff=round(secondsDiff/60,1)
    return minutesDiff


#data1=open("2018清远车站数据.csv",encoding="gbk")##2
#data1=pd.read_csv(data1)
##data2=open("2018衡阳东车站数据.csv",encoding="gbk")##8
#data2=open("2018株洲西车站数据.csv",encoding="gbk")##8
#data2=pd.read_csv(data2)
#
#def MainlineExtract(data1,data2):
#    date=pd.unique(data1["日期.y"])    
#    data5=pd.DataFrame()###本线列车集合
#    for i in date:
#        data3=data1[data1["日期.y"]==i]###清远车站的
#        data4=data2[data2["日期.y"]==i]##衡阳东
#    
#        data1NO=pd.unique(data3["车次"])
#        print(len(data1NO))
#        data2NO=pd.unique(data4["车次"])
#        TrainNo=list(set(data1NO).intersection(set(data2NO)))
#        print(i)
#    #    print(TrainNo)
#        for i in TrainNo:
#            data6=data4[data4["车次"]==i]
#            data5=pd.concat([data5,data6])
#    return data5        
#
##MainlineExtract(data1,data2).to_csv("2018衡阳东本线列车集合(新).csv",encoding='utf_8_sig')##衡阳东本线列车集合
#MainlineExtract(data1,data2).to_csv("2018株洲西本线列车集合(新).csv",encoding='utf_8_sig')##衡阳东本线列车集合
#

pd.set_option("display.width",300)
#data=open("广州北车站数据.csv",encoding="gbk")#1
#data=open("清远车站数据.csv",encoding="gbk")##2
#data=open("英德西车站数据.csv",encoding="gbk")##3
data=open("英德西车站数据（新冗余时间）.csv",encoding="gbk")##3
#data=open("韶关车站数据.csv",encoding="gbk")##4
#data=open("乐昌东车站数据.csv",encoding="gbk")##5
#data=open("郴州西车站数据.csv",encoding="gbk")##6
#data=open("耒阳西车站数据.csv",encoding="gbk")##7
#data=open("衡阳东车站数据.csv",encoding="gbk")##8
#data=open("衡山西车站数据.csv",encoding="gbk")##9
#data=open("株洲西车站数据.csv",encoding="gbk")##10
#data=open("长沙南车站数据.csv",encoding="gbk")##11

#data=open("2018广州北车站数据.csv",encoding="gbk")##1
#data=open("2018清远车站数据.csv",encoding="gbk")##2
#data=open("2018英德西车站数据.csv",encoding="gbk")##3
#data=open("2018韶关车站数据.csv",encoding="gbk")##4
#data=open("2018乐昌东车站数据.csv",encoding="gbk")##5
#data=open("2018郴州西车站数据.csv",encoding="gbk")##6
#data=open("2018耒阳西车站数据.csv",encoding="gbk")##7
#data=open("2018衡阳东车站数据.csv",encoding="gbk")##8
#data=open("2018衡山西车站数据.csv",encoding="gbk")##9
#data=open("2018株洲西车站数据.csv",encoding="gbk")##10
#data=open("2018长沙南车站数据.csv",encoding="gbk")##11



#data=open("2018衡阳东本线列车集合(新).csv",encoding="UTF-8")##8
#data=open("2018衡山西本线列车集合.csv",encoding="gbk")##9
#data=open("2018株洲西本线列车集合(新).csv",encoding="UTF-8")##10
#data=open("2018长沙南本线列车集合.csv",encoding="gbk")##11




data_2015=["广州北车站数据.csv","清远车站数据.csv","英德西车站数据.csv","韶关车站数据.csv","乐昌东车站数据.csv","郴州西车站数据.csv",
           "耒阳西车站数据.csv","衡阳东车站数据.csv","衡山西车站数据.csv","株洲西车站数据.csv","长沙南车站数据.csv"]
data_2018=["2018广州北车站数据.csv","2018清远车站数据.csv","2018英德西车站数据.csv","2018韶关车站数据.csv","2018乐昌东车站数据.csv",
           "2018郴州西车站数据.csv","2018耒阳西车站数据.csv","2018衡阳东车站数据.csv","2018衡山西车站数据.csv","2018株洲西车站数据.csv","2018长沙南车站数据.csv"]


data=pd.read_csv(data)

data["考虑停站的理想恢复影响车数"]=-100
data["以5min为间隔的理想恢复影响车数"]=-100

data["考虑停站的理想影响总时间"]=-100
data["以5min为间隔的理想影响总时间"]=-100


num=np.arange(1,data.shape[0])
data.sort_values(by=["图定到达.y"])##按计划到达排序
##计算是否停站
data.loc[data["图定到达.y"]!= data["图定出发.y"],"是否停站"]=1

#for i in np.arange(1,data.shape[0]-1):
#    #计算冗余时间
#    data.loc[i,"以5min为间隔的冗余时间"]=int(getTimeDiff(data.loc[i+1,"图定到达.y"],data.loc[i,"图定到达.y"]))-5
#    if  data.loc[i,"是否停站"]==0 or data.loc[i+1,"是否停站"]==0:##停站为1
#        data[i,"考虑停站的冗余时间"]=int(getTimeDiff(data.loc[i+1,"图定到达.y"],data.loc[i,"图定到达.y"]))-3
#    else:
#        data[i,"考虑停站的冗余时间"]=int(getTimeDiff(data.loc[i+1,"图定到达.y"],data.loc[i,"图定到达.y"]))-5
#     #计算是否使用相同的到发线   
#    print(i)  
       
print(data.shape)

data1=data.iloc[0,:].values##第一行的值
print(data1.shape)
data1=data1.reshape(1,-1)
label=data1.reshape(1,-1)##第一行的标记

col=data.columns##标题
print(type(col))
print(data1)
# print(data.head(5))
data.sort_values(by=["实际到达.y"])
# print(data.head(5))
date=pd.unique(data["日期.y"])
# print(date)
print(len(date))
data=data.drop_duplicates()
#
#for i in np.arange(1,len(date)):
#    data2=data[data.loc[:,"日期.y"]==date[i]]
#    # col=pd.DataFrame(col)
#    data2=data2.drop_duplicates()
#    data2=np.array(data2)
#    data1=np.vstack((data1,data2))
#    data1=np.vstack((data1,data.iloc[0,:].values))
#    # data1=pd.DataFrame(data1)
#    # data1=pd.concat([data1,data2])
##print(data1)
##print(data1.shape)
#data1=pd.DataFrame(data1,columns=col)


result=np.array(col).reshape(1,-1)
i=3
while i<data.shape[0]:
#while i<2000:
    if data.loc[i,"到达晚点"]>4 and data.loc[i-3:i+3,"实际间隔时间"].min()>0:###不考虑越行
        cum1=0
        cum2=0
        sum2=0
        n=0
        k=0
        j=i
        sum1=data.loc[j,"到达晚点"]
        sum2=data.loc[j,"到达晚点"]
        while data.loc[j,"到达晚点"]>cum1:
            cum1=cum1+data.loc[j+n,"考虑停站的冗余时间"]
            if data.loc[j+n,"到达晚点"]-data.loc[j+n,"考虑停站的冗余时间"]>0:
                sum1=sum1+data.loc[j+n,"到达晚点"]-data.loc[j+n,"考虑停站的冗余时间"]
            n+=1
            data.loc[i,"考虑停站的理想恢复影响车数"]=n
            data.loc[i,"考虑停站的理想影响总时间"]=sum1
        while data.loc[j,"到达晚点"]>cum2:
            cum2=cum2+data.loc[j+k,"以5min为间隔的冗余时间"]
            if data.loc[j+k,"到达晚点"]-data.loc[j+k,"以5min为间隔的冗余时间"]>0:
                sum2=sum2+data.loc[j+k,"到达晚点"]-data.loc[j+k,"以5min为间隔的冗余时间"]
            k+=1
            data.loc[i,"以5min为间隔的理想恢复影响车数"]=k
            data.loc[i,"以5min为间隔的理想影响总时间"]=sum2
        result=np.vstack((result,np.array(data.iloc[i,:])))
        m=1
        while data.loc[i+m,"到达晚点"]>0 and getTimeDiff(data.loc[i+m,"图定到达.y"],data.loc[i+m-1,"实际到达.y"])<6 :
            result=np.vstack((result,np.array(data.iloc[i+m,:]))) 
            m=m+1
        result=np.vstack((result,label))
        i=i+m
    else:      
        i=i+1
    print(i)
 
    
result[0,:] = label
print(result)
result=pd.DataFrame(result,columns=col)

'''
#i=3
#while i <data.shape[0]:
#    if data.loc[i,"到达晚点"]>4 and data.loc[i-3:i+3,"首列间隔时间"].min()>0:
#        cum=0
#        j=i
#        n=0
#        while data.loc[j,"到达晚点"]>cum:
#            cum=cum+data.loc[j+n,"冗余时间"]
#            n+=1
#            data.loc[i,"理想恢复影响车数"]=n
#        result=np.vstack((result,np.array(data.iloc[i,:])))
#        m=1
#        while data.loc[i+m,"到达晚点"]>0 and getTimeDiff(data.loc[i+m,"图定到达.y"],data.loc[i+m-1,"实际到达.y"])<6 :
#            result=np.vstack((result,np.array(data.iloc[i+m,:]))) 
#            m=m+1
#        result=np.vstack((result,label))
#        i=i+m
#    else:
#        i=i+1 
             
#result=pd.DataFrame(result)
#print(result)


result[0,:]=label
print(result)
result=pd.DataFrame(result,columns=col)
'''


j=1
for i in np.arange(0,result.shape[0]):
    if result.loc[i,"车次"]=="train":
        result.loc[i,"日期.y"]=j
        j+=1
result["影响列车数"]=-1000
result["影响总时间"]=-1000
result["列车编号"]=-1000
result["晚点持续时长"]=-1000

result["序号"]=np.arange(0,result.shape[0])
result["平均间隔时间"]=-1000
result["第1列车的晚点时间"]=0
result["第1列车后的冗余时间"]=0
result["第2列车的晚点时间"]=0
result["第3列车的晚点时间"]=0
result["第4列车的晚点时间"]=0
result["第5列车的晚点时间"]=0
result["第6列车的晚点时间"]=0

X=result[result.loc[:,"车次"]=="train"].shape[0]
result1=result.columns
result1=np.array(result1).reshape(1,-1)

for i in np.arange(1,X):
    upper=result[result.loc[:,"日期.y"]==i].loc[:,"序号"]###前行列车
    upper=np.array(upper)[-1]
#    print(type(upper))
#    print(upper)
    lower=result[result.loc[:,"日期.y"]==i+1].loc[:,"序号"]####后行列车
    lower=np.array(lower)[-1]
#    print(upper,lower)
#    if result.loc[upper+1:lower,"首列间隔时间"].min()>0:
    data3=result.iloc[upper+1:lower,:]
#    print(data3)
    if data3.loc[:,"实际间隔时间"].min()>0:##要使用按实际到达时间排序后相减的，因为不越行的话就是图定的间隔
        data3.loc[:,"影响列车数"]=data3.shape[0]
        data3.loc[:,"影响总时间"]=np.sum(data3.loc[:,"到达晚点"])
        data3.loc[:,"列车编号"]=np.arange(1,data3.shape[0]+1)
        data3.loc[:,"晚点持续时长"]=getTimeDiff(result.loc[lower-1,"实际到达.y"],result.loc[upper+1,"实际到达.y"])+data3.loc[lower-1,"到达晚点"]
        data3.loc[:,"平均间隔时间"]=np.mean(data3.loc[:,"首列间隔时间"])
        data3.index=data3["列车编号"]
#        if data3.shape[0]==1:     
#            data3.loc[:,"第1列车的晚点时间"]=data3[data3["列车编号"]==1].loc[1,"到达晚点"]
#        elif data3.shape[0]==2:
#             data3.loc[:,"第1列车的晚点时间"]=data3[data3["列车编号"]==1].loc[1,"到达晚点"]
#             data3.loc[:,"第2列车的晚点时间"]=data3[data3["列车编号"]==2].loc[2,"到达晚点"]
#        elif data3.shape[0]==3:    
#            data3.loc[:,"第1列车的晚点时间"]=data3[data3["列车编号"]==1].loc[1,"到达晚点"]
#            data3.loc[:,"第2列车的晚点时间"]=data3[data3["列车编号"]==2].loc[2,"到达晚点"]
#            data3.loc[:,"第3列车的晚点时间"]=data3[data3["列车编号"]==3].loc[3,"到达晚点"]
#        elif data3.shape[0]==4:    
#            data3.loc[:,"第1列车的晚点时间"]=data3[data3["列车编号"]==1].loc[1,"到达晚点"]
#            data3.loc[:,"第2列车的晚点时间"]=data3[data3["列车编号"]==2].loc[2,"到达晚点"]
#            data3.loc[:,"第3列车的晚点时间"]=data3[data3["列车编号"]==3].loc[3,"到达晚点"]
#            data3.loc[:,"第4列车的晚点时间"]=data3[data3["列车编号"]==4].loc[4,"到达晚点"]
#        elif data3.shape[0]==5:    
#            data3.loc[:,"第1列车的晚点时间"]=data3[data3["列车编号"]==1].loc[1,"到达晚点"]
#            data3.loc[:,"第2列车的晚点时间"]=data3[data3["列车编号"]==2].loc[2,"到达晚点"]
#            data3.loc[:,"第3列车的晚点时间"]=data3[data3["列车编号"]==3].loc[3,"到达晚点"]
#            data3.loc[:,"第4列车的晚点时间"]=data3[data3["列车编号"]==4].loc[4,"到达晚点"]
#            data3.loc[:,"第5列车的晚点时间"]=data3[data3["列车编号"]==5].loc[5,"到达晚点"]  
#        else:
#            data3.loc[:,"第1列车的晚点时间"]=data3[data3["列车编号"]==1].loc[1,"到达晚点"]
#            data3.loc[:,"第2列车的晚点时间"]=data3[data3["列车编号"]==2].loc[2,"到达晚点"]
#            data3.loc[:,"第3列车的晚点时间"]=data3[data3["列车编号"]==3].loc[3,"到达晚点"]
#            data3.loc[:,"第4列车的晚点时间"]=data3[data3["列车编号"]==4].loc[4,"到达晚点"]
#            data3.loc[:,"第5列车的晚点时间"]=data3[data3["列车编号"]==5].loc[5,"到达晚点"] 
#            data3.loc[:,"第6列车的晚点时间"]=data3[data3["列车编号"]==6].loc[6,"到达晚点"]
        data3=np.array(data3)
        result1=np.vstack((result1,data3))
    print(i)
print(result1)
result1=pd.DataFrame(result1)


#result1.to_csv("广州北车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##1
#result1.to_csv("2清远车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##2
#result1.to_csv("3英德西车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##3
#result1.to_csv("4韶关车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##4
#result1.to_csv("5乐昌东车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##5
#result1.to_csv("6郴州西车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##6
#result1.to_csv("7耒阳西车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##7
#result1.to_csv("8衡阳东车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##8
#result1.to_csv("9衡山西车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##9
#result1.to_csv("10株洲西车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##10
#result1.to_csv("11长沙南车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##11



#result1.to_csv("2018广州北车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##1
#result1.to_csv("2018清远车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##2
#result1.to_csv("2018英德西车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##3
#result1.to_csv("2018韶关车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##4
#result1.to_csv("2018乐昌东车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##5
#result1.to_csv("2018郴州西车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##6
#result1.to_csv("2018耒阳西车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##7
#result1.to_csv("2018衡阳东车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##8
#result1.to_csv("2018衡山西车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##9
#result1.to_csv("2018株洲西车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##10
#result1.to_csv("2018长沙南车站晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##11



#result1.to_csv("2018衡阳东本线列车晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##8
#result1.to_csv("2018衡山西本线列车晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##9
#result1.to_csv("2018株洲西本线列车晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##10
#result1.to_csv("2018长沙南本线列车晚点横向提取(新).csv",encoding='utf_8_sig',header=0)##10
#result1.to_csv("2018株洲西本线列车晚点横向提取(新)1.csv",encoding='utf_8_sig',header=0)##10


result1.to_csv("3英德西车站晚点横向提取(新冗余时间).csv",encoding='utf_8_sig',header=0)##3

