#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
import random
import sys,os
import time,datetime


# In[22]:


def timeStampToDate(ts):
    timeStamp = ts
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
    year = int(time.strftime("%Y",timeArray))
    month = int(time.strftime("%m",timeArray))
    date = int(time.strftime("%d",timeArray))
    day = datetime.datetime(year,month,date).strftime("%w")
    
    return month,date,day
    


# In[ ]:


def loadData(percentage):
    #打开训练集
    with open('train.csv') as trainDataFile:
        trainData = []
        trainLabel = []
        #第一行为表头，不需要，去掉
        next(trainDataFile)
        print('Loading data.....')
        #用于记录数据集大小
        count = 0
        for line in trainDataFile:
            #每读入100w数据的时候提示
            if count%1000000 == 0:
                print('Loading the '+ str(count) + ' data')
            count = count + 1
            
            #去掉每一行的换行符，并且分割字符串形成数组
            line = line.strip('\n')
            tmp = line.split(',')
            #用于存储每一行量化后的数据
            tmp_real = []
            #记录列数，对不同列进行不同操作
            column_num = 1
            #保存时间戳返回的月，日，星期几
            day = 0
            date = 0
            month = 0
            #量化每一行数据
            for item in tmp:
                #第一列行号ID不需要，去掉
                if column_num == 1:
                    column_num = column_num + 1
                    continue
                #第五列时间戳进行转化
                elif column_num == 5:
                    #通过时间戳计算月，日，星期几
                    month,date,day = timeStampToDate(int(item))
                    tmp_real.append(float(month))
                    tmp_real.append(float(date))
                    tmp_real.append(float(day))
                    column_num = column_num + 1
                    continue
                tmp_real.append(float(item))
                column_num = column_num + 1
            #将量化后的数据行插入到总数据集中
            trainData.append(tmp_real[:-1])
            #插入标签
            trainLabel.append(int(tmp_real[-1]))
        #太多的数据难以训练，使用一定的比例的训练集
        trainData = trainData[:int(len(trainData)*percentage)]
        #测试集与训练集比例为1:3
        testLen = len(trainData)/4
        trainLen = len(trainData) - testLen
        print('Data Successfully loaded' + 'We use ' + str(percentage) +'% of the total data....')
        #形成训练集和测试集合
        testData_X = []
        testData_Y = []
                             
        print('Dealing with ' + str(int(len(trainData)*percentage)) + ' data')
        print()
        #每次训练随机抽取训练集和测试集，交叉验证，可是数据集过大，操作速度太慢，注释
#         count = 0
#         for i in range(0,testLen):
#             #sys.stdout.write('Dealing with the '+ str(i) + ' data')
#             if count%500 == 0:
#                 print('Dealing with the '+ str(i) + ' data')
#             count = count + 1
#             #sys.stdout.flush()                 
#             listIndex = random.randint(0,len(trainData)-1)
#             testData_X.append(trainData[listIndex])
#             trainData.remove(trainData[listIndex])
#             testData_Y.append(trainLabel[listIndex])
#             trainLabel.remove(trainLabel[listIndex])
#         trainData_X = trainData
#         trainData_Y = trainLabel

        #形成训练集和测试集X为输入，Y为对应输出
        trainData_X = trainData[:trainLen]
        trainData_Y = trainLabel[:trainLen]
        testData_X = trainData[trainLen:]
        testData_Y = trainLabel[trainLen:]
        
        print('Data operated successfully !')
        return trainData_X,trainData_Y,testData_X,testData_Y


# In[ ]:


def main():
    #读取数据集
    trainData_X,trainData_Y,testData_X,testData_Y = loadData(0.1)
    #搭建KNN模型
    knn = KNeighborsClassifier(n_neighbors = 3)
    #使用训练集训练模型
    print('Starting KNN training process....')
    knn.fit(trainData_X,trainData_Y)
    print('Precison of training data: ' + str(knn.score(trainData_X,trainData_Y)))
    print('Precision of testing data: ' + str(knn.score(testData_X,testData_Y)))
    print('Prediction of the location : ')
    print(knn.predict(testData_X))
    
main()


# In[ ]:




