import pandas as pd

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'

def main():
    data = pd.read_csv(TEST_FILE)
    data = data.dropna(axis=0, how='any')
    data.loc[data.Embarked == 'S', 'Embarked'] = 0
    data.loc[data.Embarked == 'C', 'Embarked'] = 1
    data.loc[data.Embarked == 'Q', 'Embarked'] = 2

    # data['Sex_0'] = pd.Series()
    # data['Sex_1'] = pd.Series()
    data.loc[data.Embarked == 0, 'Sex_0'] = 0

    print(data)







if __name__ == '__main__':
    main()