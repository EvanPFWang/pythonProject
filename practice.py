if __name__ == "__main__":
    thickness = 5
    c = 'H'
    for i in range(thickness):
        print((c * i).rjust(thickness - 1) + c + (c * i).ljust(thickness - 1))

class Node:
    def __init__(self, data = None, point = None):
        self.data = data
        self.next = point

class LinkedList:
    def __init__(self):
        self.head = None

from functools import reduce
data = (100, 10, 5, 2, 2, 2, 1, 7)
print(reduce(lambda x, y: x - y, data))
print(reduce(int.__sub__, data))























def layer1(end):
    counter = 0
    for a in range(end + 1):
        for b in range(end + 1 - a):
            for c in range(end + 1 - a - b):
                for d in range(end + 1 - a - b - c):
                    for e in range(end + 1 - a - b - c - d):
                        for f in range(end + 1 - a - b - c - d - e):
                            for g in range(end + 1 - a - b - c - d - e - f):
                                for h in range(end + 1 - a - b - c - d - e - f - g):
                                    for i in range(end + 1 - a - b - c - d - e - f - g - h):
                                        counter += 1
    distcoll = []
    counter = 0
    for a in range(end + 1):
        for b in range(end + 1 - a):
            for c in range(end + 1 - a - b):
                for d in range(end + 1 - a - b - c):
                    for e in range(end + 1 - a - b - c - d):
                        for f in range(end + 1 - a - b - c - d - e):
                            for g in range(end + 1 - a - b - c - d - e - f):
                                for h in range(end + 1 - a - b - c - d - e - f - g):
                                    for i in range(end + 1 - a - b - c - d - e - f - g - h):
                                        j = end + 1 - a - b - c - d - e - f - g - h - i
                                        distcoll.append(np.array([a, b, c, d, e, f, g, h, i, j]))
    ten_k = np.zeros(10000)
    for k in countup(10000):
        ten_k[k] = stratsandavgs(np.array(distcoll))[k].coll()[0]
    return np.argmax(np.bincount(ten_k))


def layer2(end, a):
    counter = 0
    for b in range(end + 1 - a):
        for c in range(end + 1 - a - b):
            for d in range(end + 1 - a - b - c):
                for e in range(end + 1 - a - b - c - d):
                    for f in range(end + 1 - a - b - c - d - e):
                        for g in range(end + 1 - a - b - c - d - e - f):
                            for h in range(end + 1 - a - b - c - d - e - f - g):
                                for i in range(end + 1 - a - b - c - d - e - f - g - h):
                                    counter += 1
    distcoll = []
    counter = 0
    for b in range(end + 1 - a):
        for c in range(end + 1 - a - b):
            for d in range(end + 1 - a - b - c):
                for e in range(end + 1 - a - b - c - d):
                    for f in range(end + 1 - a - b - c - d - e):
                        for g in range(end + 1 - a - b - c - d - e - f):
                            for h in range(end + 1 - a - b - c - d - e - f - g):
                                for i in range(end + 1 - a - b - c - d - e - f - g - h):
                                    j = end + 1 - a - b - c - d - e - f - g - h - i
                                    distcoll.append(np.array([a, b, c, d, e, f, g, h, i, j]))
    ten_k = np.zeros(10000)
    for k in countup(10000):
        ten_k[k] = stratsandavgs(np.array(distcoll))[k].coll()[1]
    return np.argmax(np.bincount(ten_k))


def layer3(end, a, b):
    counter = 0
    for c in range(end + 1 - a - b):
        for d in range(end + 1 - a - b - c):
            for e in range(end + 1 - a - b - c - d):
                for f in range(end + 1 - a - b - c - d - e):
                    for g in range(end + 1 - a - b - c - d - e - f):
                        for h in range(end + 1 - a - b - c - d - e - f - g):
                            for i in range(end + 1 - a - b - c - d - e - f - g - h):
                                counter += 1
    distcoll = []
    counter = 0
    for c in range(end + 1 - a - b):
        for d in range(end + 1 - a - b - c):
            for e in range(end + 1 - a - b - c - d):
                for f in range(end + 1 - a - b - c - d - e):
                    for g in range(end + 1 - a - b - c - d - e - f):
                        for h in range(end + 1 - a - b - c - d - e - f - g):
                            for i in range(end + 1 - a - b - c - d - e - f - g - h):
                                j = end + 1 - a - b - c - d - e - f - g - h - i
                                distcoll.append(np.array([a, b, c, d, e, f, g, h, i, j]))
    ten_k = np.zeros(10000)
    for k in countup(10000):
        ten_k[k] = stratsandavgs(np.array(distcoll))[k].coll()[2]
    return np.argmax(np.bincount(ten_k))


def layer4(end, a, b, c):
    counter = 0
    for d in range(end + 1 - a - b - c):
        for e in range(end + 1 - a - b - c - d):
            for f in range(end + 1 - a - b - c - d - e):
                for g in range(end + 1 - a - b - c - d - e - f):
                    for h in range(end + 1 - a - b - c - d - e - f - g):
                        for i in range(end + 1 - a - b - c - d - e - f - g - h):
                            counter += 1
    distcoll = []
    counter = 0
    for d in range(end + 1 - a - b - c):
        for e in range(end + 1 - a - b - c - d):
            for f in range(end + 1 - a - b - c - d - e):
                for g in range(end + 1 - a - b - c - d - e - f):
                    for h in range(end + 1 - a - b - c - d - e - f - g):
                        for i in range(end + 1 - a - b - c - d - e - f - g - h):
                            j = end + 1 - a - b - c - d - e - f - g - h - i
                            distcoll.append(np.array([a, b, c, d, e, f, g, h, i, j]))
    ten_k = np.zeros(10000)
    for k in countup(10000):
        ten_k[k] = stratsandavgs(np.array(distcoll))[k].coll()[0]
    return np.argmax(np.bincount(ten_k))


def layer5(end, a, b, c, d):
    counter = 0
    for e in range(end + 1 - a - b - c - d):
        for f in range(end + 1 - a - b - c - d - e):
            for g in range(end + 1 - a - b - c - d - e - f):
                for h in range(end + 1 - a - b - c - d - e - f - g):
                    for i in range(end + 1 - a - b - c - d - e - f - g - h):
                        counter += 1
    distcoll = []
    counter = 0
    for e in range(end + 1 - a - b - c - d):
        for f in range(end + 1 - a - b - c - d - e):
            for g in range(end + 1 - a - b - c - d - e - f):
                for h in range(end + 1 - a - b - c - d - e - f - g):
                    for i in range(end + 1 - a - b - c - d - e - f - g - h):
                        j = end + 1 - a - b - c - d - e - f - g - h - i
                        distcoll.append(np.array([a, b, c, d, e, f, g, h, i, j]))
                        ten_k = np.zeros(10000)
    for k in countup(10000):
        ten_k[k] = stratsandavgs(np.array(distcoll))[k].coll()[0]
    return np.argmax(np.bincount(ten_k))


def layer6(end, a, b, c, d, e):
    counter = 0
    for f in range(end + 1 - a - b - c - d - e):
        for g in range(end + 1 - a - b - c - d - e - f):
            for h in range(end + 1 - a - b - c - d - e - f - g):
                for i in range(end + 1 - a - b - c - d - e - f - g - h):
                    counter += 1
    distcoll = []
    counter = 0
    for f in range(end + 1 - a - b - c - d - e):
        for g in range(end + 1 - a - b - c - d - e - f):
            for h in range(end + 1 - a - b - c - d - e - f - g):
                for i in range(end + 1 - a - b - c - d - e - f - g - h):
                    j = end + 1 - a - b - c - d - e - f - g - h - i
                    distcoll.append(np.array([a, b, c, d, e, f, g, h, i, j]))
    ten_k = np.zeros(10000)
    for k in countup(10000):
        ten_k[k] = stratsandavgs(np.array(distcoll))[k].coll()[0]
    return np.argmax(np.bincount(ten_k))


def layer7(end, a, b, c, d, e, f):
    counter = 0
    for g in range(end + 1 - a - b - c - d - e - f):
        for h in range(end + 1 - a - b - c - d - e - f - g):
            for i in range(end + 1 - a - b - c - d - e - f - g - h):
                counter += 1
    distcoll = []
    counter = 0
    for g in range(end + 1 - a - b - c - d - e - f):
        for h in range(end + 1 - a - b - c - d - e - f - g):
            for i in range(end + 1 - a - b - c - d - e - f - g - h):
                j = end + 1 - a - b - c - d - e - f - g - h - i
                distcoll.append(np.array([a, b, c, d, e, f, g, h, i, j]))
                ten_k = np.zeros(10000)
    for k in countup(10000):
        ten_k[k] = stratsandavgs(np.array(distcoll))[k].coll()[0]
    return np.argmax(np.bincount(ten_k))


def layer8(end, a, b, c, d, e, f, g):
    counter = 0
    for h in range(end + 1 - a - b - c - d - e - f - g):
        for i in range(end + 1 - a - b - c - d - e - f - g - h):
            counter += 1
    distcoll = []
    counter = 0
    for h in range(end + 1 - a - b - c - d - e - f - g):
        for i in range(end + 1 - a - b - c - d - e - f - g - h):
            j = end + 1 - a - b - c - d - e - f - g - h - i
            distcoll.append(np.array([a, b, c, d, e, f, g, h, i, j]))
    ten_k = np.zeros(10000)
    for k in countup(10000):
        ten_k[k] = stratsandavgs(np.array(distcoll))[k].coll()[0]
    return np.argmax(np.bincount(ten_k))


def layer9(end, a, b, c, d, e, f, g, h):
    counter = 0
    for i in range(end + 1 - a - b - c - d - e - f - g - h):
        counter += 1
    distcoll = []
    counter = 0
    for i in range(end + 1 - a - b - c - d - e - f - g - h):
        j = end + 1 - a - b - c - d - e - f - g - h - i
        distcoll.append(np.array([a, b, c, d, e, f, g, h, i, j]))
    ten_k = np.zeros(10000)
    for k in countup(10000):
        ten_k[k] = stratsandavgs(np.array(distcoll))[k].coll()[0]
    return np.argmax(np.bincount(ten_k))

def final(end, a, b, c, d, e, f, g, h, i):
    return end + 1 - a - b - c - d - e - f - g - h - i