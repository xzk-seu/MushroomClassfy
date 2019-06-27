import json
import os
from math import log2


class DecisionTree(object):
    def __init__(self, train_set, feature_index_set):
        """
        根据训练集构建决策树
        :param train_set:全量训练数据集
        :param feature_index_set: 所选取的特征的集合
        """
        print('construct tree, index:')
        train_index_set = [x['id'] for x in train_set]
        print('len', len(train_index_set))
        self.train_set = train_set
        self.feature_index_set = feature_index_set
        self.count_class = partition_by_class(self.train_set)
        self.majority = max(self.count_class, key=lambda x: len(self.count_class[x]))
        self.is_leaf = False
        self.best_feature = None
        self.prediction = None
        self.sub_trees = None

        if not self.is_need_to_partition():
            # 如果不需要划分，则对该节点作为一个叶子节点处理
            self.is_leaf = True
            print('不需要划分')
            print('预测值为:', self.prediction)
            print('===========================')
            return

        self.best_feature = self.get_best_feature()
        self.count_best_feature = partition_by_feature(self.train_set, self.best_feature)

        if len(self.count_best_feature) == 1:
            # 按当前特征无法划分数据集
            self.is_leaf = True
            value = self.count_best_feature.popitem()[0]
            print('按当前特征无法划分')
            print('因为当前数据在特征[%d]下取值都为%s' % (self.best_feature, value))
            self.prediction = self.majority
            print('预测值为:', self.prediction)
            print('===================================')
            return

        print('按特征[%d]对数据集划分' % self.best_feature)
        print('================================================')
        print('************************************************')

        self.sub_trees = list()
        for feature_value, index_list in self.count_best_feature.items():
            print('特征:%d，取值: %s' % (self.best_feature, feature_value))
            list(feature_index_set).remove(self.best_feature)
            new_train_set = [x for x in train_set if x['id'] in index_list]
            tree = DecisionTree(new_train_set, feature_index_set)
            temp_dict = dict(feature=self.best_feature, value=feature_value, tree=tree)
            self.sub_trees.append(temp_dict)

    def get_best_feature(self):
        info_gains = list()
        for feature in self.feature_index_set:
            info_gain = get_information_gain(self.train_set, feature)
            temp_tuple = (feature, info_gain)
            print('feature: %d, info_gain: %f' % (feature, info_gain))
            info_gains.append(temp_tuple)
        best_feature = max(info_gains, key=lambda x: x[1])[0]
        max_info_gain = -1
        for feature, info_gain in info_gains:
            if info_gain >= max_info_gain:
                max_info_gain = info_gain
                best_feature = feature
        print('当前数据下的最优特征为: %d' % best_feature)
        return best_feature

    def is_need_to_partition(self):
        """
        判断当前节点是否需要继续划分
        :return:
        """
        if len(self.train_set) == 0:
            # 当前节点对应数据集为空，不需要划分
            self.prediction = None
            print('当前节点对应数据集为空，不需要划分')
            return False
        if len(self.count_class) == 1:
            # 当前数据集包含的样本属于同一类，不需要划分
            self.prediction = self.count_class.popitem()[0]
            print('当前数据集包含的样本属于同一类，不需要划分')
            return False
        current_feature_list = self.train_set[0]['feature']
        for data in self.train_set[1:]:
            # 当前数据特征不完全一样，还需要继续划分
            if data['feature'] != current_feature_list:
                return True
            else:
                current_feature_list = data['feature']
        # 当前数据特征完全一样，无法继续划分
        print('当前数据特征完全一样，无法继续划分')
        self.prediction = self.majority
        return False

    def get_tree_dict(self):
        temp_dict = dict()
        temp_dict['best_feature'] = self.best_feature
        if self.is_leaf:
            temp_dict['prediction'] = self.prediction
        if not self.sub_trees:
            return temp_dict
        temp_dict['sub_trees'] = list()
        for tree in self.sub_trees:
            data = tree['tree'].get_tree_dict()
            temp_dict['sub_trees'].append(dict(value=tree['value'], tree=data))
        return temp_dict

    def predict(self, data):
        """
        在当前树下预测data属于哪一类
        :param data:
        :return:
        """
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        if self.is_leaf:
            print('data[%d]的预测值为:%s' % (data['id'], self.prediction))
            return self.prediction
        else:
            print('此时判断特征【%d】' % self.best_feature)
            value = data['feature'][self.best_feature]
            print('data[%d]的feature[%d]为%s' % (data['id'], self.best_feature, value))
            for tree in self.sub_trees:
                if tree['value'] == value:
                    return tree['tree'].predict(data)


def partition_by_feature(data_set, feature):
    """
        按特征对数据进行划分
        :param data_set: 按类划分data_set数据集
        :param feature: 按类划分data_set数据集
        :return: 一个字典，键为一个类名，值为该类对应的数据id
        """
    r_dict = dict()
    for data in data_set:
        f = data['feature'][feature]
        if f not in r_dict.keys():
            r_dict[f] = list()
        r_dict[f].append(data['id'])
    return r_dict


def partition_by_class(data_set):
    """
    按类对数据进行划分
    :param data_set: 按类划分data_set数据集
    :return: 一个字典，键为一个类名，值为该类对应的数据id
    """
    r_dict = dict()
    for data in data_set:
        c = data['class']
        if c not in r_dict.keys():
            r_dict[c] = list()
        r_dict[c].append(data['id'])
    return r_dict


def get_entropy(data_set):
    """
    计算data_set数据集的熵
    :param: 待计算的数据集
    :return:熵
    """
    count_class = partition_by_class(data_set)
    result = 0
    for c, c_list in count_class.items():
        freq = len(c_list) / len(data_set)
        result -= freq * log2(freq)
    return result


def get_condition_entropy(data_set, feature):
    """
    计算当前数据集在给定特征feature下的条件熵
    :param data_set: 待计算的数据集
    :param feature: 给定的特征
    :return: 条件熵
    """
    result = 0
    count_feature = partition_by_feature(data_set, feature)
    for feature_value, index_list in count_feature.items():
        # feature_value为特征的一个取值
        # index_list为该取值对应的样本id
        current_data = [x for x in data_set if x['id'] in index_list]
        freq = len(current_data) / len(data_set)
        temp_entropy = get_entropy(current_data)
        result += freq * temp_entropy
    return result


def get_information_gain(data_set, feature):
    """
    计算当前数据集在给定特征feature下的条件熵
    :param data_set: 待计算的数据集
    :param feature: 给定的特征
    :return: 信息增益
    """
    entropy = get_entropy(data_set)
    condition_entropy = get_condition_entropy(data_set, feature)
    information_gain = entropy - condition_entropy
    return information_gain


if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'data', 'data_1.json')
    d_set = json.load(open(path, 'r'))
    i_list = list(range(len(d_set)))
    test_data_set = [{'id': 0, 'class': 'p', 'feature': list('qello')},
                     {'id': 1, 'class': 'e', 'feature': list('hello')},
                     {'id': 2, 'class': 'e', 'feature': list('hello')}]
    test_index = [0, 1, 2]
    # DecisionTree(test_data_set, test_index)
    f_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    dt = DecisionTree(d_set, f_list)
    t = dt.get_tree_dict()
    json.dump(t, open('p.json', 'w'))
