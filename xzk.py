import pandas as pd
from decision_tree import DecisionTree
import json


def get_data_set():
    data_set = list()
    df = pd.read_csv('xg.csv')
    for i, row in df.iterrows():
        temp_dict = dict()
        temp_dict['id'] = row[0]
        temp_dict['class'] = row[-1]
        temp_dict['feature'] = list(row[1: -1])
        data_set.append(temp_dict)
    return data_set


if __name__ == '__main__':
    d_set = get_data_set()
    train_set = d_set[0: 10]
    test_set = d_set[10:]
    # train_set = d_set
    print(len(train_set))
    print(len(test_set))
    dt = DecisionTree(train_set, [0, 1, 2, 3, 4, 5])
    t = dt.get_tree_dict()
    json.dump(t, open('p.json', 'w'))
    correction = 0
    for data in test_set:
        if data['class'] == dt.predict(data):
            correction += 1
    accuracy = correction / len(test_set)
    print('accuracy', accuracy)
