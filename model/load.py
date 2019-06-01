import os

import pandas as pd

load_data_path = 'C:/Users/Sergey/PycharmProjects/KaggleSalePrediction/data'


def print_path_data(path: str):
    print(os.listdir(path))


load_data_train = pd.read_csv(load_data_path + '/train.tsv', header=0, sep='\t', encoding='utf-8')
load_data_test = pd.read_csv(load_data_path + '/test_no_label.tsv', header=0, sep='\t', encoding='utf-8')
load_result_sample = pd.read_csv(load_data_path + '/sample_submission.csv', header=0, sep='\t', encoding='utf-8')

if __name__ == "__main__":
    print_path_data(load_data_path)
    print('data_train:', load_data_train.shape)
    print('data_test:', load_data_test.shape)
