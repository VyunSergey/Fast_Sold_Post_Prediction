from typing import List

from sklearn.pipeline import make_union

from pipe.transform import bool_transformer, scale_transformer, encode_transformer, vector_transformer, \
    vector_tf_idf_transformer, tf_idf_transformer, tf_idf_ngram_transformer


def feature_bool_extract(features: List):
    return make_union(
        bool_transformer(features=features)
    )


def feature_numeric_scale(features: List, mean_scale_flg: bool, std_scale_flg: bool):
    return make_union(
        scale_transformer(features=features, mean_scale_flg=mean_scale_flg, std_scale_flg=std_scale_flg)
    )


def feature_categorical_encode(features: List):
    return make_union(
        encode_transformer(features=features)
    )


def feature_text_vector(feature: str):
    return make_union(
        vector_transformer(feature)
    )


def feature_text_vector_tf_idf(feature: str):
    return make_union(
        vector_tf_idf_transformer(feature)
    )


def feature_text_tf_idf(feature: str, max_features_len: int):
    return make_union(
        tf_idf_transformer(feature, max_features_len)
    )


def feature_text_tf_idf_ngram(feature: str, max_features_len: int, ngram_rng: tuple):
    return make_union(
        tf_idf_ngram_transformer(feature, max_features_len, ngram_rng)
    )
