from typing import List

from pipe.transform import bool_transformer, scale_transformer, encode_transformer, vector_transformer, \
    vector_tf_idf_transformer, tf_idf_transformer, tf_idf_ngram_transformer


# from sklearn.pipeline import make_union


def feature_bool_extract(features: List):
    return bool_transformer(features=features)


def feature_numeric_scale(features: List, mean_scale_flg: bool, std_scale_flg: bool):
    return scale_transformer(features=features, mean_scale_flg=mean_scale_flg, std_scale_flg=std_scale_flg)


def feature_categorical_encode(features: List):
    return encode_transformer(features=features)


def feature_text_vector(feature: str):
    return vector_transformer(feature)


def feature_text_vector_tf_idf(feature: str):
    return vector_tf_idf_transformer(feature)


def feature_text_tf_idf(feature: str, max_features_len: int):
    return tf_idf_transformer(feature, max_features_len)


def feature_text_tf_idf_ngram(feature: str, max_features_len: int, ngram_rng: tuple):
    return tf_idf_ngram_transformer(feature, max_features_len, ngram_rng)
