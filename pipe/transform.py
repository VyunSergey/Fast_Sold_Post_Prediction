from typing import List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer as TfidfTrn, TfidfVectorizer as TfidfVec
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder

from pipe.extract import feature_single_transformer, features_transformer


def bool_transformer(features: List):
    return features_transformer(features)


def scale_transformer(features: List, mean_scale_flg: bool, std_scale_flg: bool):
    return features_transformer(features,
                                FunctionTransformer(lambda data: data.astype(np.float64), validate=False),
                                StandardScaler(with_mean=mean_scale_flg, with_std=std_scale_flg))


def encode_transformer(features: List):
    return features_transformer(features, OneHotEncoder(categories='auto', handle_unknown='ignore'))


def vector_transformer(feature: str):
    return feature_single_transformer(feature, CountVectorizer())


def vector_tf_idf_transformer(feature: str):
    return feature_single_transformer(feature, CountVectorizer(), TfidfTrn())


def tf_idf_transformer(feature: str, max_features_len: int):
    return feature_single_transformer(feature,
                                      TfidfVec(max_features=max_features_len, token_pattern='\\w+',
                                               ngram_range=(1, 1)))


def tf_idf_ngram_transformer(feature: str, max_features_len: int, ngram_rng: tuple):
    return feature_single_transformer(feature,
                                      TfidfVec(max_features=max_features_len, token_pattern='\\w+',
                                               ngram_range=ngram_rng))
