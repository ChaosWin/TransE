from numpy import *
import operator
import numpy as np


class Test:
    # 初始化参数：（self，实体 ID list，实体向量list，关系名list， 关系向量list，训练集三元组，测试集三元组，选择标签，正负例flag）
    def __init__(self, entityList, entityVectorList, relationList, relationVectorList, tripleListTrain, tripleListTest, label="head", isFit=False):
        self.entityList = {}        # {(name: vec)}
        self.relationList = {}
        # 字典[(entity, vec)]
        for name, vec in zip(entityList, entityVectorList):
            self.entityList[name] = vec
        # 字典[(relation, vec)]
        for name, vec in zip(relationList, relationVectorList):
            self.relationList[name] = vec
        self.tripleListTrain = tripleListTrain
        self.tripleListTest = tripleListTest
        self.rank = []      # 预测结果
        self.label = label      # 选择标签
        self.isFit = isFit      # true: 如果创建的负例是正确的，则跳过

    def writeRank(self, dir):
        print("写入")
        file = open(dir, 'w')
        for r in self.rank:
            file.write(str(r[0]) + "\t")
            file.write(str(r[1]) + "\t")
            file.write(str(r[2]) + "\t")
            file.write(str(r[3]) + "\n")
        file.close()

    def getRank(self):      #
        tripletCount = 0     # 三元组的数量
        for triplet in self.tripleListTest:     # 获取每一个测试三元组
            rankList = {}   # 距离list，{(实体 ID：距离)}，获取每一个三元体在高维向量空间的距离
            for entityTemp in self.entityList.keys():      # 获取每一个实体 ID
                if self.label == "head":        # 当选择的标签是head
                    corruptedTriplet = (entityTemp, triplet[1], triplet[2])     # 构建负例的三元组
                    # 如果满足fit以及 负例实际上是正确 的，则跳过
                    if self.isFit and (corruptedTriplet in self.tripleListTrain):
                        continue
                    rankList[entityTemp] = distance(
                        self.entityList[entityTemp], self.entityList[triplet[1]], self.relationList[triplet[2]])    # 否则将实体距离放进list
                else:       # 选择的标签是tile
                    corruptedTriplet = (triplet[0], entityTemp, triplet[2])
                    if self.isFit and (corruptedTriplet in self.tripleListTrain):
                        continue
                    rankList[entityTemp] = distance(
                        self.entityList[triplet[0]], self.entityList[entityTemp], self.relationList[triplet[2]])

            # 对于每一个rank中的（ID：距离）， 按照距离进行排序
            nameRank = sorted(rankList.items(), key=operator.itemgetter(1))

            if self.label == 'head':
                numTri = 0  # head 是三元组的第1个数
            else:
                numTri = 1  # tile是三元组的第2个数
            x = 1
            for i in nameRank:
                if i[0] == triplet[numTri]:     # 如果nameRank中的第一个ID和三元组的对应实体相同，（将自身在rank中排除）则跳过
                    break
                x += 1  # 否则+1
            # 加入以下数据{(三元组，三元组的目标实体，nameRank的最近实体的ID， rank中的数量)}
            self.rank.append((triplet, triplet[numTri], nameRank[0][0], x))
            tripletCount += 1
            if triplet[1-numTri] is not nameRank[0][0]:     # 打印预测失败的rank
                print(self.rank[tripletCount-1])
            
            if tripletCount%10==0:
                print("程序还没死，别急...")

    def getRelationRank(self):
        tripletCount = 0
        self.rank = []
        for triplet in self.tripleListTest:
            rankList = {}
            for relationTemp in self.relationList.keys():
                corruptedTriplet = (triplet[0], triplet[1], relationTemp)
                if self.isFit and (corruptedTriplet in self.tripleListTrain):
                    continue
                rankList[relationTemp] = distance(
                    self.entityList[triplet[0]], self.entityList[triplet[1]], self.relationList[relationTemp])
            nameRank = sorted(rankList.items(), key=operator.itemgetter(1))
            x = 1
            for i in nameRank:
                if i[0] == triplet[2]:
                    break
                x += 1
            self.rank.append((triplet, triplet[2], nameRank[0][0], x))
            tripletCount += 1
        
    def getMeanRank(self):
        num = 0
        for r in self.rank:
            num += r[3]
        return num/len(self.rank)


def distance(h, t, r):      # 计算高维向量空间的距离，目的是找最小的前10个（越小越相似）
    h = array(h)
    t = array(t)
    r = array(r)
    s = h + r - t
    return linalg.norm(s)


def openD(dir, sp="\t"):
    #triple = (head, tail, relation)
    num = 0
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            triple = line.strip().split(sp)
            if(len(triple) < 3):      # 排除不完整的三元组
                continue
            list.append(tuple(triple))
            num += 1
    # print(num)
    return num, list       # 三元组大小和其list


def loadData(str):      # 加载向量
    fr = open(str)
    sArr = [line.strip().split("\t")
            for line in fr.readlines()]    # 每一行, [ID, value]
    datArr = [[float(s) for s in line[1][1:-1].split(", ")]
              for line in sArr]   # 每个向量表示[vec1, vec2,...,vec150]
    nameArr = [line[0] for line in sArr]    # [ID] 实体或关系的ID
    return datArr, nameArr


if __name__ == '__main__':
    dirTrain = "data/WN18/train.txt"       # 训练集
    tripleNumTrain, tripleListTrain = openD(dirTrain)       # 训练集大小和list
    print("训练集大小：", tripleNumTrain)
    dirTest = "data/WN18/test.txt"     # 测试集
    tripleNumTest, tripleListTest = openD(dirTest)      # 测试集大小和list
    print("测试集大小：", tripleNumTest)
    dirEntityVector = "result/entityVector.txt"     # 训练完的实体向量
    entityVectorList, entityList = loadData(dirEntityVector)    # 实体矩阵list, 实体 ID list
    print("实体数量：", len(entityList))
    dirRelationVector = "result/relationVector.txt"     # 训练完的关系向量
    relationVectorList, relationList = loadData(dirRelationVector)  # 关系矩阵list，关系名list
    print("关系数量：",len(relationVectorList))
    print("开始测试")

    testHeadRaw = Test(entityList, entityVectorList, relationList,
                       relationVectorList, tripleListTrain, tripleListTest)  # 生成对象
    testHeadRaw.getRank()
    print(testHeadRaw.getMeanRank())        # 距离的均值
    testHeadRaw.writeRank("result/test" + "testHeadRaw" + ".txt")
    testHeadRaw.getRelationRank()
    print(testHeadRaw.getMeanRank())
    testHeadRaw.writeRank("result/test" + "testRelationRaw" + ".txt")

    # testTailRaw = Test(entityList, entityVectorList, relationList, relationVectorList, tripleListTrain, tripleListTest, label = "tail")
    # testTailRaw.getRank()
    # print(testTailRaw.getMeanRank())
    # testTailRaw.writeRank("result/test" + "testTailRaw" + ".txt")

    # testHeadFit = Test(entityList, entityVectorList, relationList, relationVectorList, tripleListTrain, tripleListTest, isFit = True)
    # testHeadFit.getRank()
    # print(testHeadFit.getMeanRank())
    # testHeadFit.writeRank("result/test" + "testHeadFit" + ".txt")
    # testHeadFit.getRelationRank()
    # print(testHeadFit.getMeanRank())
    # testHeadFit.writeRank("result/test" + "testRelationFit" + ".txt")

    # testTailFit = Test(entityList, entityVectorList, relationList, relationVectorList, tripleListTrain, tripleListTest, isFit = True, label = "tail")
    # testTailFit.getRank()
    # print(testTailFit.getMeanRank())
    # testTailFit.writeRank("result/test" + "testTailFit" + ".txt")
