from random import uniform, sample
from numpy import *
from copy import deepcopy
import matplotlib.pyplot as plt
# import numpy as np

class TransE:
    def __init__(self, entityList, relationList, tripleList, margin = 1, learingRate = 0.05, dimension = 10, L1 = True):
        self.margin = margin    # 算法中的gamma
        self.learingRate = learingRate  # 学习率
        self.dimension = dimension  # 向量维度
        self.entityList = entityList    # 一开始，entityList是entity的list；初始化后，变为字典，key是entity，values是其向量（使用narray）。
        self.relationList = relationList    # 理由同上
        self.tripleList = tripleList    # 理由同上
        self.loss = 0   # 损失
        self.L1 = L1    # true: L1范数

    def initialize(self):
        '''
        初始化向量
        '''
        entityVectorList = {}   # 实体/向量列表
        relationVectorList = {} # 关系/向量列表
        # 对于实体列表
        for entity in self.entityList:
            n = 0   # 当前维度
            entityVector = []   # 实体向量
            while n < self.dimension:
                random = init(self.dimension)    # 初始化实体向量
                entityVector.append(random)
                n += 1
            entityVector = norm(entityVector)   # 归一化
            entityVectorList[entity] = entityVector # 字典（key实体：value向量）
        print("entityVector初始化完成，数量是%d"%len(entityVectorList))
        # 对于关系列表
        for relation in self.relationList:
            n = 0
            relationVector = [] # 关系向量
            while n < self.dimension:
                random = init(self.dimension)  # 初始化关系向量
                relationVector.append(random)
                n += 1
            relationVector = norm(relationVector)   # 归一化
            relationVectorList[relation] = relationVector   # 字典（key关系：value向量）
        print("relationVectorList初始化完成，数量是%d"%len(relationVectorList))
        # 覆盖初始化的entityList和relationList（为传入数据）
        self.entityList = entityVectorList  
        self.relationList = relationVectorList

    def transE(self, cI = 20):
        '''
        训练数据 算法
        @param cI：循环次数
        '''
        print("训练开始")
        lossList = [];
        for cycleIndex in range(cI):
            Sbatch = self.getSample(150)    # size=150的minibatch
            Tbatch = [] # 元组对（原三元组，打碎的三元组）的列表 ：{((h,r,t),(h',r,t'))}
            # 对于其中的每一个正例三元组
            for sbatch in Sbatch:  
                tripletWithCorruptedTriplet = (sbatch, self.getCorruptedTriplet(sbatch))   # tripletWithCorruptedTriplet由两个三元组组成（正例， 负例）
                if(tripletWithCorruptedTriplet not in Tbatch):  # Tbatch里没有该正负样本就加入
                    Tbatch.append(tripletWithCorruptedTriplet)  # 所以Tbatch是正负例的集合，与Sbatch不同，其为只有正例
            self.update(Tbatch)

            print("第%d次循环"%cycleIndex)
            print(self.loss)
            lossList.append(self.loss)
            self.loss = 0 # 这里清0的原因是：每次输出方便对比loss在下降
            # if cycleIndex % 100 == 0:
            #     print("第%d次循环"%cycleIndex)
            #     print(self.loss)
            #     self.writeRelationVector("result/relationVector.txt")
            #     self.writeEntilyVector("result/entityVector.txt")
            #     self.loss = 0 # 这里清0的原因是：每次输出方便对比loss在下降
        return lossList

    def getSample(self, size):
        return sample(self.tripleList, size)    # random.sample(array, count): 多个样本中取出指定数量的随机样本

    def getCorruptedTriplet(self, triplet):
        '''
        训练用随机实体替换头或尾的三元组(但不能同时替换头或尾)
        @param triplet: 正样本三元组
        @return corruptedTriplet: 负样本三元组
        '''
        i = uniform(-1, 1)
        if i < 0:   # 小于0，更换头实体
            while True:
                entityTemp = sample(self.entityList.keys(), 1)[0]   # 在所有实体集中取出一个实体，实体存在列表中，所以需要[0], for example ['/m/04bdlg']
                if entityTemp != triplet[0]:    # 防止取出的头实体和我们要更换的头实体相同
                    break
            corruptedTriplet = (entityTemp, triplet[1], triplet[2]) # 生成负样本
        else:   # 大于等于0，更换尾实体
            while True:
                entityTemp = sample(self.entityList.keys(), 1)[0]   # 在所有实体集中取出一个实体作为替换掉尾实体的实体
                if entityTemp != triplet[1]:
                    break
            corruptedTriplet = (triplet[0], entityTemp, triplet[2]) # 生成负样本
        return corruptedTriplet

    def update(self, Tbatch):
        '''
        按梯度下降法对词向量参数进行更新
        '''
        copyEntityList = deepcopy(self.entityList)  # 深拷贝，不改变原元素
        copyRelationList = deepcopy(self.relationList)
        
        for tripletWithCorruptedTriplet in Tbatch:
            headEntityVector = copyEntityList[tripletWithCorruptedTriplet[0][0]]    # tripletWithCorruptedTriplet是原三元组和打碎的三元组的元组tuple
            tailEntityVector = copyEntityList[tripletWithCorruptedTriplet[0][1]]
            relationVector = copyRelationList[tripletWithCorruptedTriplet[0][2]]
            headEntityVectorWithCorruptedTriplet = copyEntityList[tripletWithCorruptedTriplet[1][0]]
            tailEntityVectorWithCorruptedTriplet = copyEntityList[tripletWithCorruptedTriplet[1][1]]
            
            headEntityVectorBeforeBatch = self.entityList[tripletWithCorruptedTriplet[0][0]]    # tripletWithCorruptedTriplet是原三元组和打碎的三元组的元组tuple
            tailEntityVectorBeforeBatch = self.entityList[tripletWithCorruptedTriplet[0][1]]
            relationVectorBeforeBatch = self.relationList[tripletWithCorruptedTriplet[0][2]]
            headEntityVectorWithCorruptedTripletBeforeBatch = self.entityList[tripletWithCorruptedTriplet[1][0]]
            tailEntityVectorWithCorruptedTripletBeforeBatch = self.entityList[tripletWithCorruptedTriplet[1][1]]
            
            if self.L1: # l1范数
                distTriplet = distanceL1(headEntityVectorBeforeBatch, tailEntityVectorBeforeBatch, relationVectorBeforeBatch)           
                distCorruptedTriplet = distanceL1(headEntityVectorWithCorruptedTripletBeforeBatch, tailEntityVectorWithCorruptedTripletBeforeBatch ,  relationVectorBeforeBatch)    # 关系没有变
            else:   # l2范数
                distTriplet = distanceL2(headEntityVectorBeforeBatch, tailEntityVectorBeforeBatch, relationVectorBeforeBatch)
                distCorruptedTriplet = distanceL2(headEntityVectorWithCorruptedTripletBeforeBatch, tailEntityVectorWithCorruptedTripletBeforeBatch ,  relationVectorBeforeBatch)
            eg = self.margin + distTriplet - distCorruptedTriplet      # 合页损失函数的内部函数
            if eg > 0:  # [function]+ 是一个取正值的函数
                self.loss += eg # 累加损失
                if self.L1:
                    tempPositive = 2 * self.learingRate * (tailEntityVectorBeforeBatch - headEntityVectorBeforeBatch - relationVectorBeforeBatch)
                    tempNegtative = 2 * self.learingRate * (tailEntityVectorWithCorruptedTripletBeforeBatch - headEntityVectorWithCorruptedTripletBeforeBatch - relationVectorBeforeBatch)                
                    tempPositiveL1 = []
                    tempNegtativeL1 = []
                    for i in range(self.dimension):
                        if tempPositive[i] >= 0:    # 第i维大于0则在列表中增加1，否则为-1
                            tempPositiveL1.append(1)
                        else:
                            tempPositiveL1.append(-1)   
                        if tempNegtative[i] >= 0:   # 同上
                            tempNegtativeL1.append(1)
                        else:
                            tempNegtativeL1.append(-1)
                    tempPositive = array(tempPositiveL1)  # 列表转换为数组，方便矩阵运算
                    tempNegtative = array(tempNegtativeL1)
                else:
                    # 这里是梯度
                    tempPositive = 2 * self.learingRate * (tailEntityVectorBeforeBatch - headEntityVectorBeforeBatch - relationVectorBeforeBatch)
                    tempNegtative = 2 * self.learingRate * (tailEntityVectorWithCorruptedTripletBeforeBatch - headEntityVectorWithCorruptedTripletBeforeBatch - relationVectorBeforeBatch)
    
                headEntityVector = headEntityVector + tempPositive
                tailEntityVector = tailEntityVector - tempPositive
                relationVector = relationVector + tempPositive - tempNegtative
                headEntityVectorWithCorruptedTriplet = headEntityVectorWithCorruptedTriplet - tempNegtative
                tailEntityVectorWithCorruptedTriplet = tailEntityVectorWithCorruptedTriplet + tempNegtative

                # 只归一化这几个刚更新的向量，而不是按原论文那些一口气全更新了
                copyEntityList[tripletWithCorruptedTriplet[0][0]] = norm(headEntityVector)
                copyEntityList[tripletWithCorruptedTriplet[0][1]] = norm(tailEntityVector)
                copyRelationList[tripletWithCorruptedTriplet[0][2]] = norm(relationVector)
                copyEntityList[tripletWithCorruptedTriplet[1][0]] = norm(headEntityVectorWithCorruptedTriplet)
                copyEntityList[tripletWithCorruptedTriplet[1][1]] = norm(tailEntityVectorWithCorruptedTriplet)
                
        self.entityList = copyEntityList
        self.relationList = copyRelationList
        
    def writeEntilyVector(self, dir):
        print("写入实体")
        entityVectorFile = open(dir, 'w')
        for entity in self.entityList.keys():
            entityVectorFile.write(entity+"\t")
            entityVectorFile.write(str(self.entityList[entity].tolist()))
            entityVectorFile.write("\n")
        entityVectorFile.close()

    def writeRelationVector(self, dir):
        print("写入关系")
        relationVectorFile = open(dir, 'w')
        for relation in self.relationList.keys():
            relationVectorFile.write(relation + "\t")
            relationVectorFile.write(str(self.relationList[relation].tolist()))
            relationVectorFile.write("\n")
        relationVectorFile.close()

def init(dimension):
    return uniform(-6/(dimension**0.5), 6/(dimension**0.5)) # 对于每个向量的每个维度在 正负6/sqrt(dimension) 范围内随机生成

def distanceL1(h, t ,r):
    s = h + r - t
    sum = fabs(s).sum()
    return sum

def distanceL2(h, t, r):
    s = h + r - t
    sum = (s*s).sum()
    return sum
 
def norm(list):
    '''
    归一化
    @param 向量
    @return: 向量的平方和的开方后的向量
    '''
    var = linalg.norm(list)     # numpy.linalg.norm(list, ord=None, axis=None, keepdims=False)，默认为l2范数
    i = 0
    while i < len(list):
        list[i] = list[i]/var
        i += 1
    return array(list)  # 返回array

def openDetailsAndId(dir,sp="\t"):
    idNum = 0
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            DetailsAndId = line.strip().split(sp)
            list.append(DetailsAndId[0])
            idNum += 1
    return idNum, list

def openTrain(dir,sp="\t"):
    num = 0
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            triple = line.strip().split(sp)
            if(len(triple)<3):  # 跳过无效三元组
                continue
            list.append(tuple(triple))
            num += 1
    return num, list

def draw(list):
    x = linspace(1,len(list),len(list))
    plt.plot(x, list)
    plt.show()


if __name__ == '__main__':
    # 读取实体集合 txt文本格式为：实体\ID，切分
    dirEntity = "data/FB15k/entity2id.txt"
    entityIdNum, entityList = openDetailsAndId(dirEntity)
    # 读取关系集合 txt文本格式为：关系\ID，切分
    dirRelation = "data/FB15k/relation2id.txt"
    relationIdNum, relationList = openDetailsAndId(dirRelation)
    # 读取训练集 txt文本格式为：实体\实体\ID，切分
    dirTrain = "data/FB15k/train.txt"
    tripleNum, tripleList = openTrain(dirTrain)
    print("打开TransE")
    transE = TransE(entityList,relationList,tripleList, margin=1, dimension = 100, L1=False)
    print("TranE初始化")
    transE.initialize()
    # transE.transE(1500)
    lossList = transE.transE(10000)
    # 绘图
    draw(lossList)

    transE.writeRelationVector("result/relationVector.txt")
    transE.writeEntilyVector("result/entityVector.txt")

