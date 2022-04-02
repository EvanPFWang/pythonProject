# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from typing import List

import numpy as np


# Press the green button in the gutter to run the script.


class Pair:
    pass

    def __init__(self, fAvg, allocation):
        self.average = fAvg
        self.allocation = allocation

    def alloc(self):
        return self.allocation

    def fAvg(self):
        return self.average

    def __str__(self):
        return '{self.average}: {self.allocation}'.format(self=self)


def crazyincubator(size):
    allstrats = np.random.dirichlet([1, 1, 1, 1, 1], size)*100
    lOver = 100 - np.sum(allstrats, axis=1)
    for i in range(size):
        for a in lOver:
            for b in range(int(np.rint(a))):
                allstrats[i, np.random.randint(0, 10)] += 1
        np.random.shuffle(allstrats[i])
    return allstrats


def strategicincubator(alphas,size):
    allstrats = np.rint(np.random.dirichlet(alphas,size))*100
    lOver = 100 - np.sum(allstrats,axis=1)
    for i in range(size):
        for a in lOver:
            for b in range(int(np.rint(a))):
                allstrats[i, np.random.randint(0, 5)] += 1
    return allstrats


def getScore(strat1, strat2):
    result = strat1 - strat2
    score1 = 0
    for i in range(5):
        if result[i] > 0:
            if i >= 2:
                if result[i - 1] > 0 and result[i - 2] > 0:
                    return score1 + sum(range(i, 5))
            score1 += (i + 1)
    return score1

def avgScore(strat1, competition):
    fullScore = 0
    for strat2 in competition:
        fullScore += getScore(strat1, strat2)
    return fullScore / np.shape(competition)[0]


def topTen(collection):
    list = []
    for i in range(10):
        list.append(stratsandavgs(collection)[i].alloc())
    return np.array(list)


def countup(high):
    num = 0
    while num < high:
        yield num
        num += 1

def stratsandavgs(collection1, collection2) -> List[Pair]:
    repos = []
    for i in countup(collection1.shape[0]):
        repos.append(Pair(np.rint(10000 * avgScore(collection1[i], collection2)*collection2.shape[0] / 10000 / (collection2.shape[0]+1)), collection1[i]))
    repos.sort(key=Pair.fAvg, reverse=True)
    return repos


def stratsandavgs(collection) -> List[Pair]:
    repos = []
    for i in countup(collection.shape[0]):
        repos.append(Pair(np.rint(10000 * avgScore(collection[i], collection)*collection.shape[0] / 10000 / (collection.shape[0]+1)), collection[i]))
    repos.sort(key=Pair.fAvg, reverse=True)
    return repos
    # (repos, key=lambda x: x.avg, reverse=True)

#based off of every other possibility's averages calculated, the layer functions are combined in the everypossible() function.
def layer1(end):
    counter = 0
    for a in range(end):
        for b in range(end - a):
            for c in range(end - a - b):
                for d in range(end - a - b - c):
                    counter += 1
    ten_k = np.zeros(500)
    distcoll = []
    for a in range(end):
        for b in range(end - a):
            for c in range(end - a - b):
                for d in range(end - a - b - c):
                    e = end - a - b - c - d
                    print(np.array([a, b, c, d, e]))
                    distcoll.append(np.array([a, b, c, d, e]))
    orderedcoll = stratsandavgs(np.array(distcoll))
    print("finished")
    for k in countup(500):
        ten_k[k] = orderedcoll[k].alloc()[0]
    return np.argmax(np.bincount(ten_k))


def layer2(end, a):
    counter = 0
    for b in range(end - a):
        for c in range(end - a - b):
            for d in range(end - a - b - c):
                counter += 1
    distcoll = []
    ten_k = np.zeros(500)
    for b in range(end - a):
        for c in range(end - a - b):
            for d in range(end - a - b - c):
                e = end - a - b - c - d
                distcoll.append(np.array([a, b, c, d, e]))
    orderedcoll = stratsandavgs(np.array(distcoll))
    print("finished")
    for k in countup(500):
        ten_k[k] = orderedcoll[k].alloc()[0]
    return np.argmax(np.bincount(ten_k))


def layer3(end, a, b):
    counter = 0
    for c in range(end - a - b):
        for d in range(end - a - b - c):
            counter += 1
    distcoll = []
    ten_k = np.zeros(500)
    for c in range(end - a - b):
        for d in range(end - a - b - c):
            e = end - a - b - c - d
            distcoll.append(np.array([a, b, c, d, e]))
    orderedcoll = stratsandavgs(np.array(distcoll))
    print("finished")
    for k in countup(500):
        ten_k[k] = orderedcoll[k].alloc()[0]
    return np.argmax(np.bincount(ten_k))


def layer4(end, a, b, c):
    counter = 0
    for d in range(end - a - b - c):
        counter += 1
    distcoll = []
    ten_k = np.zeros(500)
    for d in range(end - a - b - c):
        e = end - a - b - c - d
        distcoll.append(np.array([a, b, c, d, e]))
    orderedcoll = stratsandavgs(np.array(distcoll))
    print("finished")
    for k in countup(500):
        ten_k[k] = orderedcoll[k].alloc()[0]
    return np.argmax(np.bincount(ten_k))


def layer5(end, a, b, c, d):
    return end - a - b - c - d


def everypossible(end):
    one = layer1(end)
    two = layer2(end, one)
    three = layer3(end, one, two)
    four = layer4(end, one, two, three)
    five = layer5(end, one, two, three, four)

    return np.array(one, two, three, four, five)*100

def montecarlo(size):
    return topTen(crazyincubator(size))


if __name__ == '__main__':
    import numpy as np
    print(everypossible(20))
    import tensorflow as tf
    from random import choices
    # montecarloAssumptions = {
    #     "b123": np.array([31, 31, 31, 1, 1, 1, 1, 1, 1, 1]),
    #     "b123c": np.array([1, 1, 44, 8, 7, 8, 7, 8, 8, 8]),
    #     "mc1": np.array([1, 31, 31, 31, 1, 1, 1, 1, 1, 1]),
    #     "mc2": np.array([1, 1, 31, 31, 31, 1, 1, 1, 1, 1]),
    #     "SS1": np.array([1, 31, 1, 1, 31, 1, 1, 31, 1, 1]),
    #     "SS2": np.array([1, 1, 31, 1, 1, 31, 1, 31, 1, 1]),
    #     "SS3": np.array([1, 1, 31, 1, 31, 1, 1, 31, 1, 1]),
    #     "SS4": np.array([1, 1, 31, 1, 1, 31, 1, 1, 31, 1]),
    #     "R1": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    #     "R2": np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])}
    # strat = ["b123", "b123c","mc1", "mc2", "SS1", "SS2", "SS3", "SS4", "R1", "R2"]
    # dist = [0.18,0.13, 0.13, 0.13, 0.02, 0.02, 0.02, 0.01, 0.15]
    #
    # mc_sample = 1000
    # b123 = strategicincubator(montecarloAssumptions["b123"], int(dist[0] * mc_sample))
    # b123c = strategicincubator(montecarloAssumptions["b123c"], int(dist[1] * mc_sample))
    # mc1 = strategicincubator(montecarloAssumptions["mc1"], int(dist[2] * mc_sample))
    # mc2 = strategicincubator(montecarloAssumptions["mc2"], int(dist[3] * mc_sample))
    # mc2 = strategicincubator(montecarloAssumptions["mc2"], int(dist[3] * mc_sample))
    # ss1 = strategicincubator(montecarloAssumptions["SS1"], int(dist[4] * mc_sample))
    # ss2 = strategicincubator(montecarloAssumptions["SS2"], int(dist[5] * mc_sample))
    # ss3 = strategicincubator(montecarloAssumptions["SS3"], int(dist[6] * mc_sample))
    # ss4 = strategicincubator(montecarloAssumptions["SS4"], int(dist[7] * mc_sample))
    # r1 = strategicincubator(montecarloAssumptions["R1"], int(dist[8] * mc_sample))
    #
    # montecarlotry = np.concatenate(b123, b123c, mc1)
    # montecarlotry2 = np.concatenate(mc2, ss1, ss2)
    # montecarlotry3 = np.concatenate(ss3, ss4, r1)
    #
    # montecarlotryfinal = np.concatenate(montecarlotry,montecarlotry2, montecarlotry3)
    #
    # distcoll = []
    # for a in range(20):
    #     for b in range(20 - a):
    #         for c in range(20 - a - b):
    #             for d in range(20 - a - b - c):
    #                 e = 20 - a - b - c - d
    #                 distcoll.append(5*np.array([a, b, c, d, e, f, g, h, i, j]))
    #     print("since I was not able to get my other algorithm to work, here is a consolation answer")
    #     print(stratsandavgs(np.array(distcoll), montecarlotryfinal)[0].alloc()[0])

    # stratalloc = {
    #     "b123": np.array([31, 31, 31, 1, 1, 1, 1, 1, 1, 1]),
    #     "b123c": np.array([0, 0, 44, 8, 8, 8, 8, 8, 8, 8]),
    #     "mc1": np.array([1, 31, 31, 31, 1, 1, 1, 1, 1, 1]),
    #     "mc2": np.array([1, 1, 31, 31, 31, 1, 1, 1, 1, 1]),
    #     "bfda": everypossible(5),
    #     "SS1": np.array([1, 31, 1, 1, 31, 1, 1, 31, 1, 1]),
    #     "SS2": np.array([1, 1, 31, 1, 1, 31, 1, 31, 1, 1]),
    #     "SS3": np.array([1, 1, 31, 1, 31, 1, 1, 31, 1, 1]),
    #     "SS4": np.array([1, 1, 31, 1, 1, 31, 1, 1, 31, 1]),
    #     "R1": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    #     "R2": np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])}
    # strat = ["b123", "b123c","mc1", "mc2", "bfda", "SS1", "SS2", "SS3", "SS4", "R1", "R2"]
    # dist = [0.18,0.13, 0.13, 0.13, 0.11, 0.02, 0.02, 0.02, 0.01, 0.1, 0.05]

    #full_Sample = 1000
    #b123 = strategicincubator(stratalloc["b123"],dist[0]*full_Sample)
    #b123c = strategicincubator(stratalloc["b123c"],dist[1]*full_Sample)
    #mc1 = strategicincubator(stratalloc["mc1"],dist[2]*full_Sample)
    #mc2 = strategicincubator(stratalloc["mc2"],dist[3]*full_Sample)
    #bfda = strategicincubator(stratalloc["bfda"], dist[4] * full_Sample)
    #ss1 = strategicincubator(stratalloc["SS1"],dist[5]*full_Sample)
    #ss2 = strategicincubator(stratalloc["SS2"],dist[6]*full_Sample)
    #ss3 = strategicincubator(stratalloc["SS3"],dist[7]*full_Sample)
    #ss4 = strategicincubator(stratalloc["SS4"],dist[8]*full_Sample)
    #r1 = strategicincubator(stratalloc["R1"],dist[9]*full_Sample)
    #r2 = []

    #for i in countup(dist[10]*full_Sample):
    #    r2.append(np.array([10,10,10,10,10,10,10,10,10,10]))
    #r2 = np.array(r2)



    #monte = []
    #for i in range(10):
    #    monte.append(stratsandavgs(crazyincubator(1000))[i].alloc())# find top ten in sample size of 100,000
    #monte = np.array(monte)
    #if np.array_equal(monte[0], stratsandavgs(monte)[0].alloc()):
    #    mc = monte[0]
    #else:
    #    mc = stratsandavgs(monte)[0].alloc()

    #full_test = np.concatenate(b123, b123c, mc1, mc2, bfda, ss1, ss2, ss3, ss4, r1, r2, axis=0)

