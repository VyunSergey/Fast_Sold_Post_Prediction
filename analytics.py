from typing import List

import numpy as np
from sklearn.pipeline import FeatureUnion

from model.load import load_data_train, load_model
from model.transform import data_columns, target_columns, feature_pre_processor, feature_columns
from pipe.process import feature_bool_extract, feature_numeric_scale, feature_categorical_encode, feature_text_vector, \
    feature_text_tf_idf, feature_text_tf_idf_ngram
from utils.time import timer


def analyze(name: str, model, feature_names: List):
    feature_imp = model.feature_importances_
    # feature_data = np.array(sorted(zip(feature_names, feature_imp), key=lambda x: x[1]))
    print('Model ' + name + ' feature importance shape:', feature_imp.shape)
    print('Model ' + name + ' feature importance top 10:', sorted(feature_imp)[:10])
    print('Model ' + name + ' feature importance zero count:', len([x for x in feature_imp if x == 0]))
    print('Model ' + name + ' feature importance:', feature_imp)


if __name__ == "__main__":
    with timer('data loading'):
        data_all_train = load_data_train[data_columns].copy()
        y_all_target = load_data_train[target_columns].copy()

    with timer('data pre_processing'):
        data_train_prep = feature_pre_processor(data_all_train, feature_columns)

    with timer('data processing'):
        data_processor = FeatureUnion([
            ('feature_bool_extract', feature_bool_extract(features=['delivery_available', 'payment_available'])),
            ('feature_numeric_scale', feature_numeric_scale(features=['price', 'lat', 'long', 'img_num'],
                                                            mean_scale_flg=True, std_scale_flg=True)),
            ('feature_categorical_encode', feature_categorical_encode(features=['product_type', 'category_id',
                                                                                'subcategory_id', 'sold_mode'])),
            ('feature_text_vector_region', feature_text_vector(feature='region')),
            ('feature_text_vector_city', feature_text_vector(feature='city')),
            ('feature_text_tf_idf_name_text', feature_text_tf_idf(feature='name_text', max_features_len=70000)),
            ('feature_text_tf_idf_ngram_desc_text', feature_text_tf_idf_ngram(feature='desc_text',
                                                                              max_features_len=400000,
                                                                              ngram_rng=(1, 2)))
        ])

        data_processor1 = feature_numeric_scale(features=['price', 'lat', 'long', 'img_num'],
                                                mean_scale_flg=True, std_scale_flg=True)

        data_train = data_processor.fit_transform(data_train_prep).astype(np.float32)
        data_train1 = data_processor1.fit_transform(data_train_prep).astype(np.float32)

    with timer('model loading'):
        model_name = 'GradientBoostingClassifier_Mon_Jun__3_20-36-37_2019'
        model = load_model(model_name)
        print('data_processor: ', data_processor1)
        # feature_names = data_processor.get_feature_names()
        feature_names = []

    with timer('model analyzing'):
        analyze(model_name, model, feature_names)
