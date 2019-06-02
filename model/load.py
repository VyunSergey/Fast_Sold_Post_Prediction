import os
import pickle as pkl

import pandas as pd

from model.save import save_model_path

load_data_path = 'C:/Users/Sergey/PycharmProjects/Fast_Sold_Post_Prediction/data'
load_model_path = save_model_path


def print_path_data(path: str):
    print(os.listdir(path))


def load_data(data_name: str, header: int = 0, sep: str = '\t', encoding: str = 'utf-8') -> pd.DataFrame:
    return pd.read_csv(load_data_path + '/' + data_name, header=header, sep=sep, encoding=encoding)


def load_model(model_name: str):
    file_name = model_name + '.pkl'
    with open(load_model_path + '/' + file_name, mode='rb') as file:
        model = pkl.load(file)
    return model


load_data_train = load_data('train.tsv')
load_data_test = load_data('test_no_label.tsv')
load_result_sample = load_data('sample_submission.csv')

if __name__ == "__main__":
    print_path_data(load_data_path)
    print('data train:', load_data_train.shape)
    print('data test:', load_data_test.shape)

    test_model_name = 'DecisionTreeClassifier_Sun_Jun__2_07-48-57_2019'
    test_model = load_model(test_model_name)
    print('test model:', test_model)
