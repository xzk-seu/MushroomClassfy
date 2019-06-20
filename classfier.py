import pandas as pd


if __name__ == '__main__':
    fr = pd.read_csv('mushrooms.csv')
    
    for index, row in fr.iterrows():
        print(row)
        if index == 10:
            break
