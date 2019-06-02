import os
import pickle as pkl
import time

import numpy as np
import pandas as pd

from utils.lists import list_to_str
from utils.strings import regexp_replace

save_model_path = 'C:/Users/Sergey/PycharmProjects/Fast_Sold_Post_Prediction/result/models'
save_score_path = 'C:/Users/Sergey/PycharmProjects/Fast_Sold_Post_Prediction/result/scores'
save_prediction_path = 'C:/Users/Sergey/PycharmProjects/Fast_Sold_Post_Prediction/result/predictions'


def print_path_data(desc: str, path: str):
    print(desc, os.listdir(path))


def time_format(time_string: str) -> str:
    return regexp_replace(':', '-', regexp_replace(' ', '_', time_string))


def save_model(name: str, model):
    now_time_formatted = time_format(str(time.asctime(time.localtime(time.time()))))
    file_name = name + '_' + now_time_formatted + '.pkl'
    with open(save_model_path + '/' + file_name, mode='wb') as file:
        pkl.dump(model, file)


def save_score(name: str, score: float):
    now_time_formatted = time_format(str(time.asctime(time.localtime(time.time()))))
    file_name = name + '_' + now_time_formatted + '.txt'
    score_text = 'score: ' + str(score)
    with open(save_score_path + '/' + file_name, mode='w', encoding='utf-8') as file:
        file.write(score_text)


def save_prediction(name: str, prediction: pd.DataFrame):
    now_time_formatted = time_format(str(time.asctime(time.localtime(time.time()))))
    file_name = name + '_' + now_time_formatted + '.csv'
    prediction['text'] = prediction['product_id'].astype(str) + ',' + prediction['score'].astype(str)
    header_text = 'product_id,score'
    prediction_text = list_to_str('\n', [header_text] + list(prediction['text'].values.flatten()))
    with open(save_prediction_path + '/' + file_name, mode='w', encoding='utf-8') as file:
        file.write(prediction_text + '\n')


if __name__ == "__main__":
    print_path_data('Save Models Path:', save_model_path)
    print_path_data('Save Scores Path:', save_score_path)
    print_path_data('Save Predictions Path:', save_prediction_path)
    save_model('test', np.array(['test']))
    save_score('test', 1.0)
    save_prediction('test', pd.DataFrame(
        np.array([['test_00', 0.1], ['test_10', 1.1]]), columns=['product_id', 'score']))
