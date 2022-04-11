# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from typing import List
game_size = 1000;
troops = 100;
castles = 10;
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

#uses alphas to return a set of strategies that is "size" large
def strategicincubator(alphas,size):
    allstrats = np.rint(np.random.dirichlet(alphas,size)*troops)
    l_Over = troops - np.sum(allstrats, axis=1)
#edited the method to use more noise to normalize each strat
    for strat_ind in range(size):
        allstrats[strat_ind, 0] += l_Over[strat_ind]
        np.random.shuffle(allstrats[strat_ind])
    return allstrats


def getScore(strat1, strat2):
    result = strat1 - strat2
    score1 = 0
    consecutive = 0
    for i in range(castles):
        if result[i] > 0:
            #if i >= 2:
            #    if result[i - 1] > 0 and result[i - 2] > 0:
            #        return score1 + sum(range(i, 10))
            score1 += (i + 1)
            consecutive += 1
            if consecutive <= 0:
                consecutive = 1
                continue
            elif consecutive < 3:
                continue
            else:
                return score1 - (i+1) + sum(range(i+1, castles))
        elif result[i] < 0:
            consecutive -= 1
            if consecutive >= 0:
                consecutive = -1
                continue
            elif consecutive > -3:
                continue
            else:
                return score1
        elif result[i] == 0:
            consecutive == 0
    return score1

#competition is a set of strategies
def fullScore(strat1, competition):
    fullScore = 0
    for strat2 in competition:
        fullScore += getScore(strat1, strat2)
    return fullScore


def topTen(collection):
    list = []
    for i in range(castles):
        list.append(stratsandavgs(collection)[i].alloc())
    return np.array(list)


def countup(high):
    num = 0
    while num < high:
        yield num
        num += 1

#Test all of collection1 against collection2
#Yield: sorted repos of Pairs with highest favg
def stratsandavgs(collection1, collection2) -> List[Pair]:
    repos = [collection1.shape[0]]
    for i in countup(collection1.shape[0]):
        repos[i]=(Pair(np.rint(fullScore(collection1[i], collection2) / (collection2.shape[0])), collection1[i]))
    repos.sort(key=Pair.fAvg, reverse=True)
    return repos


def stratsandavgs(collection) -> List[Pair]:
    #pairdt = np.dtype([np.array([10,10,10,10,10,10,10,10,10,10]),('avg_score',np.float64)])
    repos = [collection.shape[0]-1]
    for i in countup(collection.shape[0]):
        repos.append(Pair(np.rint(10000 * fullScore(collection[i], collection)/ 10000 / (collection.shape[0]+1)), collection[i]))
        repos[i] = (Pair(np.rint(fullScore(collection[i], collection) / (collection.shape[0]-1)), collection[i]))

    repos.sort(key=Pair.fAvg, reverse=True)
    return repos
    # (repos, key=lambda x: x.avg, reverse=True)

#based off of every other possibility's averages calculated, the layer functions are combined in the everypossible() function.
def layer1(end):
    counter = 0
    ten_k = np.zeros(500)
    distcoll = []
    for a in range(end):
        for b in range(end - a):
            for c in range(end - a - b):
                for d in range(end - a - b - c):
                    for e in range(end - a - b - c - d):
                        for f in range(end - a - b - c - d - e):
                            for g in range(end - a - b - c - d - e - f):
                                for h in range(end - a - b - c - d - e - f - g):
                                    for i in range(end - a - b - c - d - e - f - g - h):
                                        counter += 1
                                        j = end - a - b - c - d - e - f - g - h - i
                                        print(np.array([a, b, c, d, e, f, g, h, i, j]))
                                        distcoll.append(np.array([a, b, c, d, e, f, g, h, i, j]))
    for k in countup(500):
        ten_k[k] = stratsandavgs(np.array(distcoll))[k].alloc()[0]
    return np.argmax(np.bincount(ten_k))


def layer2(end, a):
    counter = 0
    distcoll = []
    ten_k = np.zeros(500)
    for b in range(end - a):
        for c in range(end - a - b):
            for d in range(end - a - b - c):
                for e in range(end - a - b - c - d):
                    for f in range(end - a - b - c - d - e):
                        for g in range(end - a - b - c - d - e - f):
                            for h in range(end - a - b - c - d - e - f - g):
                                for i in range(end - a - b - c - d - e - f - g - h):
                                    j = end - a - b - c - d - e - f - g - h - i
                                    distcoll.append(np.array([a, b, c, d, e, f, g, h, i, j]))
    for k in countup(500):
        ten_k[k] = stratsandavgs(np.array(distcoll))[k].alloc()[1]
    return np.argmax(np.bincount(ten_k))


def layer3(end, a, b):
    counter = 0

    distcoll = []
    ten_k = np.zeros(500)
    for c in range(end - a - b):
        for d in range(end - a - b - c):
            for e in range(end - a - b - c - d):
                for f in range(end - a - b - c - d - e):
                    for g in range(end - a - b - c - d - e - f):
                        for h in range(end - a - b - c - d - e - f - g):
                            for i in range(end - a - b - c - d - e - f - g - h):
                                counter += 1
                                j = end - a - b - c - d - e - f - g - h - i
                                distcoll.append(np.array([a, b, c, d, e, f, g, h, i, j]))
    for k in countup(500):
        ten_k[k] = stratsandavgs(np.array(distcoll))[k].alloc()[2]
    return np.argmax(np.bincount(ten_k))


def layer4(end, a, b, c):
    counter = 0

    distcoll = []
    ten_k = np.zeros(500)
    for d in range(end - a - b - c):
        for e in range(end - a - b - c - d):
            for f in range(end - a - b - c - d - e):
                for g in range(end - a - b - c - d - e - f):
                    for h in range(end - a - b - c - d - e - f - g):
                        for i in range(end - a - b - c - d - e - f - g - h):
                            counter += 1
                            j = end - a - b - c - d - e - f - g - h - i
                            distcoll.append(np.array([a, b, c, d, e, f, g, h, i, j]))
    for k in countup(500):
        ten_k[k] = stratsandavgs(np.array(distcoll))[k].alloc()[0]
    return np.argmax(np.bincount(ten_k))


def layer5(end, a, b, c, d):
    counter = 0
    distcoll = []
    ten_k = np.zeros(500)
    for e in range(end - a - b - c - d):
        for f in range(end - a - b - c - d - e):
            for g in range(end - a - b - c - d - e - f):
                for h in range(end - a - b - c - d - e - f - g):
                    for i in range(end - a - b - c - d - e - f - g - h):
                        counter += 1
                        j = end - a - b - c - d - e - f - g - h - i
                        distcoll.append(np.array([a, b, c, d, e, f, g, h, i, j]))
    for k in countup(500):
        ten_k[k] = stratsandavgs(np.array(distcoll))[k].alloc()[0]
    return np.argmax(np.bincount(ten_k))


def layer6(end, a, b, c, d, e):
    counter = 0
    distcoll = []
    ten_k = np.zeros(500)
    for f in range(end - a - b - c - d - e):
        for g in range(end - a - b - c - d - e - f):
            for h in range(end - a - b - c - d - e - f - g):
                for i in range(end - a - b - c - d - e - f - g - h):
                    counter += 1
                    j = end - a - b - c - d - e - f - g - h - i
                    distcoll.append(np.array([a, b, c, d, e, f, g, h, i, j]))
    for k in countup(500):
        ten_k[k] = stratsandavgs(np.array(distcoll))[k].alloc()[0]
    return np.argmax(np.bincount(ten_k))


def layer7(end, a, b, c, d, e, f):
    counter = 0
    distcoll = []
    ten_k = np.zeros(500)
    for g in range(end - a - b - c - d - e - f):
        for h in range(end - a - b - c - d - e - f - g):
            for i in range(end - a - b - c - d - e - f - g - h):
                counter += 1
                j = end - a - b - c - d - e - f - g - h - i
                distcoll.append(np.array([a, b, c, d, e, f, g, h, i, j]))
    for k in countup(500):
        ten_k[k] = stratsandavgs(np.array(distcoll))[k].alloc()[0]
    return np.argmax(np.bincount(ten_k))


def layer8(end, a, b, c, d, e, f, g):
    counter = 0
    for h in range(end - a - b - c - d - e - f - g):
        for i in range(end - a - b - c - d - e - f - g - h):
            counter += 1
    distcoll = []
    ten_k = np.zeros(500)
    for h in range(end - a - b - c - d - e - f - g):
        for i in range(end - a - b - c - d - e - f - g - h):
            j = end - a - b - c - d - e - f - g - h - i
            distcoll.append(np.array([a, b, c, d, e, f, g, h, i, j]))
    for k in countup(500):
        ten_k[k] = stratsandavgs(np.array(distcoll))[k].alloc()[0]
    return np.argmax(np.bincount(ten_k))


def layer9(end, a, b, c, d, e, f, g, h):
    counter = 0
    distcoll = []
    ten_k = np.zeros(500)
    for i in range(end - a - b - c - d - e - f - g - h):
        counter += 1
        j = end - a - b - c - d - e - f - g - h - i
        distcoll.append(np.array([a, b, c, d, e, f, g, h, i, j]))
    for k in countup(500):
        ten_k[k] = stratsandavgs(np.array(distcoll))[k].alloc()[0]
    return np.argmax(np.bincount(ten_k))

def final(end, a, b, c, d, e, f, g, h, i):
    return end - a - b - c - d - e - f - g - h - i


def everypossible(end):
    one = layer1(end)
    two = layer2(end, one)
    three = layer3(end, one, two)
    four = layer4(end, one, two, three)
    five = layer5(end, one, two, three, four)
    six = layer6(end, one, two, three, four, five)
    seven = layer7(end, one, two, three, four, five, six)
    eight = layer8(end, one, two, three, four, five, six, seven)
    nine = layer9(end, one, two, three, four, five, six, seven, eight)
    ten = final(end, one, two, three, four, five, six, seven, eight, nine)

    return np.random.dirichlet(np.array(one, two, three, four, five, six, seven, eight, nine, ten), 1)[0]*100

def montecarlo(size):
    return topTen(strategicincubator(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),size));

if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf
    from random import choices
    full_Sample = game_size

    montecarloAssumptions = {
        "b123": np.array([31, 31, 31, 1, 1, 1, 1, 1, 1, 1]),
        "b123c": np.array([0, 0, 44, 8, 8, 8, 8, 8, 8, 8]),
        "mc1": np.array([1, 31, 31, 31, 1, 1, 1, 1, 1, 1]),
        "mc2": np.array([1, 1, 31, 31, 31, 1, 1, 1, 1, 1]),
        "SS1": np.array([1, 31, 1, 1, 31, 1, 1, 31, 1, 1]),
        "SS2": np.array([1, 1, 31, 1, 1, 31, 1, 31, 1, 1]),
        "SS3": np.array([1, 1, 31, 1, 31, 1, 1, 31, 1, 1]),
        "SS4": np.array([1, 1, 31, 1, 1, 31, 1, 1, 31, 1]),
        "R1": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        "R2": np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])}
    strat = ["b123", "b123c","mc1", "mc2", "SS1", "SS2", "SS3", "SS4", "R1", "R2"]
    dist = [0.18,0.13, 0.13, 0.13, 0.02, 0.02, 0.02, 0.01, 0.1, 0.05]

    mc_sample = game_size
    b123 = strategicincubator(montecarloAssumptions["b123"], dist[0] * full_Sample)
    b123c = strategicincubator(montecarloAssumptions["b123c"], dist[1] * full_Sample)
    mc1 = strategicincubator(montecarloAssumptions["mc1"], dist[2] * full_Sample)
    mc2 = strategicincubator(montecarloAssumptions["mc2"], dist[3] * full_Sample)
    ss1 = strategicincubator(montecarloAssumptions["SS1"], dist[4] * full_Sample)
    ss2 = strategicincubator(montecarloAssumptions["SS2"], dist[5] * full_Sample)
    ss3 = strategicincubator(montecarloAssumptions["SS3"], dist[6] * full_Sample)
    ss4 = strategicincubator(montecarloAssumptions["SS4"], dist[7] * full_Sample)
    r1 = strategicincubator(montecarloAssumptions["R1"], dist[8] * full_Sample)
    r2 = []

    for i in countup(dist[10] * full_Sample):
        r2.append(np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]))
        r2 = np.array(r2)
        montecarlotry = np.concatenate(b123, b123c, mc1, mc2, ss1, ss2, ss3, ss4, r1, r2, axis=0)

        distcoll = []
        for a in range(troops):
            for b in range(troops - a):
                for c in range(troops - a - b):
                    for d in range(troops - a - b - c):
                        for e in range(troops - a - b - c - d):
                            for f in range(troops - a - b - c - d - e):
                                for g in range(troops - a - b - c - d - e - f):
                                    for h in range(troops - a - b - c - d - e - f - g):
                                        for i in range(troops - a - b - c - d - e - f - g - h):
                                            j = troops - a - b - c - d - e - f - g - h - i
                                            print(np.array([a, b, c, d, e, f, g, h, i, j]))
                                            distcoll.append(np.array([a, b, c, d, e, f, g, h, i, j]))
        print("since I was not able to get my other algorithm to work, here is a consolation answer")
        print(stratsandavgs(np.array(distcoll), montecarlotry)[0].alloc()[0])

    stratalloc = {
        "b123": np.array([31, 31, 31, 1, 1, 1, 1, 1, 1, 1]),
        "b123c": np.array([0, 0, 44, 8, 8, 8, 8, 8, 8, 8]),
        "mc1": np.array([1, 31, 31, 31, 1, 1, 1, 1, 1, 1]),
        "mc2": np.array([1, 1, 31, 31, 31, 1, 1, 1, 1, 1]),
        "bfda": everypossible(5),
        "SS1": np.array([1, 31, 1, 1, 31, 1, 1, 31, 1, 1]),
        "SS2": np.array([1, 1, 31, 1, 1, 31, 1, 31, 1, 1]),
        "SS3": np.array([1, 1, 31, 1, 31, 1, 1, 31, 1, 1]),
        "SS4": np.array([1, 1, 31, 1, 1, 31, 1, 1, 31, 1]),
        "R1": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        "R2": np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])}
    strat = ["b123", "b123c","mc1", "mc2", "bfda", "SS1", "SS2", "SS3", "SS4", "R1", "R2"]
    dist = [0.18,0.13, 0.13, 0.13, 0.11, 0.02, 0.02, 0.02, 0.01, 0.1, 0.05]

    b123 = strategicincubator(stratalloc["b123"],dist[0]*full_Sample)
    b123c = strategicincubator(stratalloc["b123c"],dist[1]*full_Sample)
    mc1 = strategicincubator(stratalloc["mc1"],dist[2]*full_Sample)
    mc2 = strategicincubator(stratalloc["mc2"],dist[3]*full_Sample)
    bfda = strategicincubator(stratalloc["bfda"], dist[4] * full_Sample)
    ss1 = strategicincubator(stratalloc["SS1"],dist[5]*full_Sample)
    ss2 = strategicincubator(stratalloc["SS2"],dist[6]*full_Sample)
    ss3 = strategicincubator(stratalloc["SS3"],dist[7]*full_Sample)
    ss4 = strategicincubator(stratalloc["SS4"],dist[8]*full_Sample)
    r1 = strategicincubator(stratalloc["R1"],dist[9]*full_Sample)
    r2 = []

    for i in countup(dist[10]*full_Sample):
        r2.append(np.array([10,10,10,10,10,10,10,10,10,10]))
    r2 = np.array(r2)



    #monte = []
    #for i in range(10):
    #    monte.append(stratsandavgs(crazyincubator(1000))[i].alloc())# find top ten in sample sizesize of 100,000
    #monte = np.array(monte)
    #if np.array_equal(monte[0], stratsandavgs(monte)[0].alloc()):
    #    mc = monte[0]
    #else:
    #    mc = stratsandavgs(monte)[0].alloc()

    full_test = np.concatenate(b123, b123c, mc1, mc2, bfda, ss1, ss2, ss3, ss4, r1, r2, axis=0)

