import os
import json
from decision_tree import DecisionTree


if __name__ == '__main__':
    batch = list()
    for i in range(10):
        path = os.path.join(os.getcwd(), 'data', 'data_%d.json' % i)
        batch.append(json.load(open(path, 'r')))
    feature_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

    a_list = list()
    for train_set in batch:
        dt = DecisionTree(train_set, feature_list)
        t = dt.get_tree_dict()
        test_set = batch[-1]
        correction = 0
        for data in test_set:
            if data['class'] == dt.predict(data):
                correction += 1
        accuracy = correction / len(test_set)
        print('accuracy', accuracy)
        a_list.append(accuracy)

    print(a_list)

    # json.dump(t, open('p.json', 'w'))
    #
    # path = os.path.join(os.getcwd(), 'data', 'data_%d.json' % 9)
    # test_set = json.load(open(path, 'r'))
    # correction = 0
    # for data in test_set:
    #     if data['class'] == dt.predict(data):
    #         correction += 1
    # accuracy = correction/len(test_set)
    # print('accuracy', accuracy)
