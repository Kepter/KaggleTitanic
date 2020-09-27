import math
import operator
import heapq

class KNN:
    def __init__(self):
        super().__init__()

    def euclidean_distance(self, a, b):
        dist = 0
        for i in range(min(len(a), len(b))):
            dist += pow(a[i] - b[i], 2)
        
        return math.sqrt(dist)

    def fit(self, x_train, y_train):
        self.data = x_train
        self.labels = y_train

    def predict(self, x, k=3):
        distances = []
        for i in range(len(self.data)):
            distances.append((self.euclidean_distance(x, self.data[i]), self.labels[i]))

        distances.sort()
        votes = dict()
        for i in range(k):
            sqdist = pow(distances[i][0], 2)
            if (sqdist == 0):
                sqdist = 0.00001

            if distances[i][1] in votes:
                votes[distances[i][1]] += (1 / sqdist)
            else:
                votes[distances[i][1]] = (1 / sqdist)

        votes = sorted(votes, key=votes.get, reverse=True)
        return votes[0]

