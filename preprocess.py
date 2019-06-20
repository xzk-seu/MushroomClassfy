import pandas as pd
import json
import os


if __name__ == '__main__':
    fr = pd.read_csv('mushrooms.csv')
    data_list = list()
    for index, row in fr.iterrows():
        temp_dict = {'id': index, 'class': row['class'], 'feature': list(row[1:])}
        data_list.append(temp_dict)
    json.dump(data_list, open('total.json', 'w'))

    data_path = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # 将数据集分为10分，用于交叉验证
    step = len(data_list) // 10
    for i in range(10):
        start = i * step
        if i == 9:
            temp_data = data_list[start:]
        else:
            temp_data = data_list[start: start+step]
        path = os.path.join(data_path, 'data_%d.json' % i)
        json.dump(temp_data, open(path, 'w'))
        print('%d /%d' % (len(temp_data), len(data_list)))
