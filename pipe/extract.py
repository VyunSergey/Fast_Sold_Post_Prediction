from operator import itemgetter
from typing import List

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer


def feature_single_getter(feature: str):
    return FunctionTransformer(itemgetter(feature), validate=False)


def features_getter(features: List):
    return FunctionTransformer(itemgetter(features), validate=False)


def feature_single_transformer(feature: str, *vec) -> Pipeline:
    return make_pipeline(feature_single_getter(feature), *vec)


def features_transformer(features: List, *vec) -> Pipeline:
    return make_pipeline(features_getter(features), *vec)
