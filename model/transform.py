from typing import List

import pandas as pd

from model.load import load_data_train, load_data_test
from utils.constants import russian_conjunctions, russian_prepositions, russian_pronouns
from utils.lists import list_drop_na, list_equals, list_contains
from utils.strings import str_drop_digits, str_drop_trash, str_drop_spec, str_drop_len, str_deduplicate
from utils.time import timer

data_columns = [
    'product_id', 'product_type', 'date_created', 'region', 'city', 'category_id', 'subcategory_id',
    'owner_id', 'name_text', 'desc_text', 'price', 'delivery_available', 'payment_available',
    'img_num', 'lat', 'long', 'properties', 'sold_mode']

target_columns = ['sold_fast']

feature_columns = [
    'product_type', 'region', 'city', 'category_id', 'subcategory_id',
    'name_text', 'desc_text', 'price', 'delivery_available', 'payment_available',
    'img_num', 'lat', 'long', 'sold_mode']

data_train = load_data_train[data_columns].copy()
data_test = load_data_test[data_columns].copy()
y_target = load_data_train[target_columns].copy()

# print(type(data_train[['name_text']]))


def text_column_normalizing(df: pd.DataFrame, column: str) -> pd.Series:
    df[column] = df[column].fillna('').map(lambda x: x.strip().lower())
    return df[column]


def text_column_clearing(df: pd.DataFrame, column: str) -> pd.Series:
    df[column] = df[column].map(lambda x: str_drop_spec(str_drop_digits(x)))
    df[column] = df[column].map(lambda x: str_drop_len(' ', x, 3))
    df[column] = df[column].map(lambda x: str_deduplicate(' ', x))
    return pd.Series(list_drop_na(df[column].values.flatten()))


def text_column_drop_trash(df: pd.DataFrame, column: str, trash_words: List) -> pd.Series:
    df[column] = df[column].map(lambda x: str_drop_trash(' ', x, trash_words))
    return df[column]


def feature_pre_processor(df: pd.DataFrame, features_cols: List) -> pd.DataFrame:
    df_columns = df.columns
    drop_columns = list(set(df_columns).difference(set(features_cols)))
    df = df.drop(columns=drop_columns)

    rus_conj = pd.DataFrame(russian_conjunctions, columns=['conj'])
    rus_conj['conj'] = text_column_normalizing(rus_conj, 'conj')
    rus_conj['conj'] = text_column_clearing(rus_conj, 'conj')
    rus_conj = list_drop_na(list(rus_conj['conj'].values.flatten()))

    rus_preps = pd.DataFrame(russian_prepositions, columns=['preps'])
    rus_preps['preps'] = text_column_normalizing(rus_preps, 'preps')
    rus_preps['preps'] = text_column_clearing(rus_preps, 'preps')
    rus_preps = list_drop_na(list(rus_preps['preps'].values.flatten()))

    rus_pron = pd.DataFrame(russian_pronouns, columns=['pron'])
    rus_pron['pron'] = text_column_normalizing(rus_pron, 'pron')
    rus_pron['pron'] = text_column_clearing(rus_pron, 'pron')
    rus_pron = list_drop_na(list(rus_pron['pron'].values.flatten()))

    trash_words = list(set(rus_conj + rus_preps + rus_pron))

    intersect_columns = list(set(df.columns).intersection(set(features_cols)))

    if not list_equals(intersect_columns, features_cols):
        print("WARNING: New Columns of DataFrame NOT EQUALS to Feature Columns",
              "New Columns:", df.columns,
              "Feature Columns:", features_cols,
              sep='\n')

    if list_contains(intersect_columns, ['region']):
        df['region'] = text_column_normalizing(df, 'region')
        df['region'] = text_column_clearing(df, 'region')

    if list_contains(intersect_columns, ['city']):
        df['city'] = text_column_normalizing(df, 'city')
        df['city'] = text_column_clearing(df, 'city')

    if list_contains(intersect_columns, ['name_text']):
        df['name_text'] = text_column_normalizing(df, 'name_text')
        df['name_text'] = text_column_clearing(df, 'name_text')
        df['name_text'] = text_column_drop_trash(df, 'name_text', trash_words)

    if list_contains(intersect_columns, ['region', 'city', 'name_text', 'desc_text']):
        df['desc_text'] = text_column_normalizing(df, 'region') + ' ' + \
                          text_column_normalizing(df, 'city') + ' ' + \
                          text_column_normalizing(df, 'name_text') + ' ' + \
                          text_column_normalizing(df, 'desc_text')

    elif list_contains(intersect_columns, ['city', 'name_text', 'desc_text']):
        df['desc_text'] = text_column_normalizing(df, 'city') + ' ' + \
                          text_column_normalizing(df, 'name_text') + ' ' + \
                          text_column_normalizing(df, 'desc_text')

    elif list_contains(intersect_columns, ['name_text', 'desc_text']):
        df['desc_text'] = text_column_normalizing(df, 'name_text') + ' ' + \
                          text_column_normalizing(df, 'desc_text')

    elif list_contains(intersect_columns, ['desc_text']):
        df['desc_text'] = text_column_normalizing(df, 'desc_text')

    if list_contains(intersect_columns, ['desc_text']):
        df['desc_text'] = text_column_clearing(df, 'desc_text')
        df['desc_text'] = text_column_drop_trash(df, 'desc_text', trash_words)

    if list_contains(intersect_columns, ['delivery_available']):
        df['delivery_available'] = df['delivery_available'].astype('int32')

    if list_contains(intersect_columns, ['payment_available']):
        df['payment_available'] = df['payment_available'].astype('int32')

    return df


if __name__ == "__main__":
    with timer('data pre_processing'):
        data_train_prep = feature_pre_processor(load_data_train, feature_columns)
        print('data_train_prep:', data_train_prep.shape)
