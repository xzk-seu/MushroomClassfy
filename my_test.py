import os
import json
from decision_tree import DecisionTree


if __name__ == '__main__':
    train_set = list()
    for i in range(9):
        path = os.path.join(os.getcwd(), 'data', 'data_%d.json' % i)
        train_set.extend(json.load(open(path, 'r')))
    feature_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    dt = DecisionTree(train_set, feature_list)
