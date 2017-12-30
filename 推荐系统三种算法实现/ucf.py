#coding:utf-8

import random
import math
from  numpy import *
import csv
import datetime

NumOfUsers=1000

def GetData(datafile='u.data'):
    '''
    把datafile文件中数据读出来，返回data对象
    :param datafile: 数据源文件名称
    :return: 一个列表，每一个元素是一个元组(userId,movieId)
    '''
    data=[]
    try:
        file=open(datafile)
    except:
        print ("No such file name"+datafile)
    for line in file:
        line=line.split('\t')
        try:
            data.append((int(line[0]),int(line[1])))
        except:
            pass
    file.close()
    return data

def SplitData(data,M,k,seed):
    '''
    划分训练集和测试集
    :param data:传入的数据
    :param M:测试集占比
    :param k:一个任意的数字，用来随机筛选测试集和训练集
    :param seed:随机数种子，在seed一样的情况下，其产生的随机数不变
    :return:train:训练集 test：测试集，都是字典，key是用户id,value是电影id集合
    '''
    test=dict()
    train=dict()
    random.seed(seed)
    # 在M次实验里面我们需要相同的随机数种子，这样生成的随机序列是相同的
    for user,item in data:
        if random.randint(0,M)!=k:
            # 相等的概率是1/M，所以M决定了测试集在所有数据中的比例
            # 选用不同的k就会选定不同的训练集和测试集
            if user not in test.keys():
                test[user]=set()
            test[user].add(item)
        else:
            if user not in train.keys():
                train[user]=set()
            train[user].add(item)
    return train,test

def Recall(train,test,N,k):
    '''

    :param train: 训练集
    :param test: 测试集
    :param N: TopN推荐中N数目
    :param k:
    :return:返回召回率
    '''
    hit=0# 预测准确的数目
    totla=0# 所有行为总数
    W,relatedusers=ImprovedCosineSimilarity(train)
    for user in train.keys():
        tu=test[user]
        rank=GetRecommendation(user,train,N,k,W,relatedusers)
        for item in rank:
            if item in tu:
                hit+=1
        totla+=len(tu)
    return hit/(totla*1.0)

def Precision(train,test,N,k):
    '''

    :param train:
    :param test:
    :param N:
    :param k:
    :return:
    '''
    hit=0
    total=0
    W, relatedusers = ImprovedCosineSimilarity(train)
    for user in train.keys():
        tu = test[user]
        rank = GetRecommendation(user, train, N, k, W, relatedusers)
        for item in rank:
            if item in tu:
                hit += 1
        total += N
    return hit / (total * 1.0)

def Coverage(train,test,N,k):
    '''
    计算覆盖率
    :param train:训练集 字典user->items
    :param test: 测试机 字典 user->items
    :param N: topN推荐中N
    :param k:
    :return:覆盖率
    '''
    recommend_items=set()
    all_items=set()
    W,relatedusers=ImprovedCosineSimilarity(train)
    for user in train.keys():
        for item in train[user]:
            all_items.add(item)
        rank=GetRecommendation(user,train,N,k,W,relatedusers)
        for item in rank:
            recommend_items.add(item)
    return len(recommend_items)/(len(all_items)*1.0)


def Popularity(train,test,N,k):
    '''
    计算平均流行度
    :param train:训练集 字典user->items
    :param test: 测试机 字典 user->items
    :param N: topN推荐中N
    :param k:
    :return:覆盖率
    '''
    item_popularity=dict()
    W,relatedusers=ImprovedCosineSimilarity(train)
    for user,items in train.items():
        for item in items:
            if item not in item_popularity:
                item_popularity[item]=0
            item_popularity[item]+=1
    ret=0
    n=0
    for user in train.keys():
        rank= GetRecommendation(user, train, N, k, W, relatedusers)
        for item in rank:
            if item!=0:
                ret+=math.log(1+item_popularity[item])
                n+=1
    ret/=(n*1.0)
    return ret

def CosineSimilarty(train):
    '''
       计算训练集中每两个用户的余弦相似度
       这个函数没有实际价值，复杂度相当高，而且容易Out Of Memory，即在训练集大的时候容易产生内存不足的错误
       但是这个函数比较容易看出公式的原型，可以借此理解公式运用
       :param train: 训练集,字典user->items
       :return: 返回相似度矩阵
    '''
    W=dict()
    print (len(train.keys()))
    for u in train.keys():
        for v in train.keys():
            if u==v:
                continue
            W[(u,v)]=len(train[u]&train[v])
            W[(u,v)]/=math.sqrt(len(train[u])*len(train[v])*1.0)
            W[(v,u)]=W[(u,v)]
    return W

def ImprovedCosineSimilarity(train):
    '''
    计算用户相似度
    :param train:
    :return: 返回用户相似度矩阵W,W[u][v]表示u,v的相似度
    :return: 返回相关用户user_relatedusers字典，key为用户id,value为和而用户有共同电影的用户集合。
    '''
    #建立电影->用户倒排表
    item_user=dict()
    for u,items in train.items():
        for i in items:
            if i not in item_user:
                item_user[i]=set()
            item_user[i].add(u)
    #C[u][v] 表示用户u和用户v之间共同喜欢的电影
    C=zeros([NumOfUsers,NumOfUsers],dtype=float16)
    #N[u]表示u评价的电影数目
    N=zeros([NumOfUsers],dtype=int32)
    # user_relatedusers[u]表示u的相关用户(共同电影不为零的用户)
    user_relatedusers=dict()
    # 对于每个电影，把它对应的用户组合C[u][v]加一
    for item,users in item_user.items():
        for u in users:
            N[u]+=1
            for v in users:
                if u==v:
                    continue
                if u not in user_relatedusers:
                    user_relatedusers[u]=set()
                user_relatedusers[u].add(v)
                C[u][v]+=(1/math.log(1+len(users)))
    #用户相似度矩阵
    W=zeros([NumOfUsers,NumOfUsers],dtype=float16)
    for u in range(1,NumOfUsers):
        if u in user_relatedusers:
            for v in user_relatedusers[u]:
                W[u][v]=C[u][v]/sqrt(N[u]*N[v])

    return W,user_relatedusers

def Recommend(user,train,W,relatedusers,k,N):
    '''
    通过相似度矩阵W得到和user相似的rank字典
    :param user:用户id
    :param train: 训练集
    :param W: 相似度矩阵
    :param relatedusers:
    :param k: 决定了从相似用户中取出多少进行计算
    :param N:
    :return: rank字典，包含了所有兴趣程度不为0的电影，按照从大到小排序
    '''
    rank=dict()
    for i in range(1,1700):
        rank[i]=0# i表示user可能喜欢的电影id，初始兴趣程度为0
    k_users=dict()
    try:
        for v in relatedusers[user]:
            k_users[v]=W[user][v]
    except KeyError:
        print ("User "+str(user)+" doesn't have any related users in train set")

    k_users=sorted(k_users.items(),key=lambda x:x[1],reverse=True)
    k_users=k_users[0:k]#取前k个用户

    for i in range(1700):
        for v,wuv in k_users:
            if i in train[v] and i not in train[user]:#取出被user相似用户v产生行为的电影，同时user没有和这部电影产生行为
                rank[i]+=wuv*1

    return sorted(rank.items(),key=lambda d:d[1],reverse=True)

def GetRecommendation(user,train,N,k,W,relatedusers):
    '''
     获得N个推荐
     :param user: 用户
     :param train: 训练集
     :param W: 相似度矩阵
     :param N: 推荐数目N
     :param k: 决定了从相似用户中取出多少个进行计算
     :return: recommend字典，key是movie id，value是兴趣程度
     '''
    rank=Recommend(user,train,W,relatedusers,k,N)
    recommend=dict()
    for i in range(N):
        recommend[rank[i][0]]=rank[i][1]
    return recommend


def evaluate(train,test,N,k):
    ##计算一系列评测标准

    recommends=dict()
    W,relatedusers=ImprovedCosineSimilarity(train)
    for user in test:
        recommends[user]=GetRecommendation(user,train,N,k,W,relatedusers)

    recall=Recall(train,test,N,k)
    precision=Precision(train,test,N,k)
    coverage=Coverage(train,test,N,k)
    popularity=Popularity(train,test,N,k)
    return recall,precision,coverage,popularity

def test1():
    data=GetData()
    train,test=SplitData(data,2,1,1)
    del data
    user=int(input("Input the user id \n"))
    print("The train set contains the movies of the user: ")
    print(train[user])
    N=int(input("Input the number of recommendations\n"))
    k=int(input("Input the number of related users\n"))

    starttime=datetime.datetime.now()
    W,relatedusers=ImprovedCosineSimilarity(train)
    endtime=datetime.datetime.now()
    print("it takes ",(endtime-starttime).seconds," seconds to get W")

    starttime=datetime.datetime.now()
    recommend=GetRecommendation(user,train,N,k,W,relatedusers)
    endtime=datetime.datetime.now()
    print("it takes ",(endtime-starttime).seconds," seconds to get recommend for one user")

    W,relatedusers=ImprovedCosineSimilarity(train)
    recommend=GetRecommendation(user,train,N,k,W,relatedusers)
    print(recommend)
    for item in recommend:
        print(item),
        if(item in test[user]):
            print("  True")
        else:
            print("  False")

def test2():
    N=int(input("Input the number of recommendations: \n"))
    k=int(input("Input the number of related users: \n"))
    data = GetData()
    train, test = SplitData(data, 2, 1, 1)
    del data
    recall,precision,coverage,popularity=evaluate(train,test,N,k)
    print("Recall: ",recall)
    print("Precision: ",precision)
    print("Coverage: ",coverage)
    print("Popularity: ",popularity)

if __name__=='__main__':
    test2()
