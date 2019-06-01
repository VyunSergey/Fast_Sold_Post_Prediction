import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.pipeline import make_union

# from model.classifiers.linear import linear, linear_name, linear_params_grid
from model.classifiers.decision_tree import tree, tree_name, tree_params_grid
from model.load import load_data_train, load_data_test
from model.save import save_model, save_score, save_prediction
from model.transform import data_columns, target_columns, feature_pre_processor, feature_columns
from pipe.process import feature_bool_extract, feature_numeric_scale, feature_categorical_encode, feature_text_vector, \
    feature_text_tf_idf, feature_text_tf_idf_ngram
from utils.time import timer


# from model.classifiers.gradient_boosting import gbc, gbc_name, gbc_params_grid


def model_fit(name, model, params_grid, score, X, y):
    x_train, y_train = X, y
    cross_val = KFold(n_splits=10, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=params_grid, scoring=score, cv=cross_val)

    grid_search.fit(x_train, y_train)
    print('GridSearchCV ' + name + ' best score:', grid_search.best_score_)
    print('GridSearchCV ' + name + ' best params:', grid_search.best_params_)
    best_model = grid_search.best_estimator_
    return best_model


def model_validate(name, model, score, X, y):
    x_valid, y_valid = X, y

    # print('Target y_valid:', y_valid)
    # print('Model predict:', model.predict(x_valid))
    # print('Model predict_proba:', model.predict_proba(x_valid))
    # print('Model predict_proba 0:',
    # list(model.predict_proba(x_valid)[:, 0].astype(np.float32).reshape(1, -1).flatten()))
    # print('Model predict_proba 1:',
    # list(model.predict_proba(x_valid)[:, 1].astype(np.float32).reshape(1, -1).flatten()))
    y_predict = model.predict(x_valid).astype(np.int32).reshape(1, -1).flatten()
    y_predict_proba = model.predict_proba(x_valid)[:, 1].astype(np.int32).reshape(1, -1).flatten()
    model_score = score(list(y_valid), list(y_predict))
    model_score_proba = score(list(y_valid), list(y_predict_proba))
    print('Model ' + name + ' valid predict score:', model_score)
    print('Model ' + name + ' valid predict_proba score:', model_score_proba)
    return model_score


def model_submit(model, X):
    x_test = X
    return model.predict_proba(x_test)[:, 1].astype(np.float32).reshape(1, -1).flatten()


def main():
    with timer('data loading'):
        data_all_train = load_data_train[data_columns].copy()
        data_all_test = load_data_test[data_columns].copy()
        y_all_target = load_data_train[target_columns].copy()

    with timer('data pre_processing'):
        data_train_prep = feature_pre_processor(data_all_train, feature_columns)
        data_test_prep = feature_pre_processor(data_all_test, feature_columns)

    with timer('data processing'):
        data_processor = make_union(
            feature_bool_extract(features=['delivery_available', 'payment_available']),
            feature_numeric_scale(features=['price', 'lat', 'long', 'img_num'],
                                  mean_scale_flg=True, std_scale_flg=True),
            feature_categorical_encode(features=['product_type', 'category_id', 'subcategory_id', 'sold_mode']),
            feature_text_vector(feature='region'),
            feature_text_vector(feature='city'),
            feature_text_tf_idf(feature='name_text', max_features_len=70000),
            feature_text_tf_idf_ngram(feature='desc_text', max_features_len=400000, ngram_rng=(1, 2))
        )

        data_train = data_processor.fit_transform(data_train_prep).astype(np.float32)
        data_test = data_processor.transform(data_test_prep).astype(np.float32)
        y_target = y_all_target.astype(np.float32).values.reshape(1, -1).flatten()

        print('data_train: {0} of {1}'.format(data_train.shape, data_train.dtype))
        print('data_test: {0} of {1}'.format(data_test.shape, data_test.dtype))

        (X_train, X_valid, y_train, y_valid) = train_test_split(data_train, y_target, test_size=0.20, random_state=42)
        print('X_train: {0} of {1}'.format(X_train.shape, X_train.dtype))
        print('X_valid: {0} of {1}'.format(X_valid.shape, data_train.dtype))

    with timer('data fit'):
        name, model, params_grid = tree_name, tree, tree_params_grid
        # name, model, params_grid = gbc_name, gbc, gbc_params_grid
        score = make_scorer(roc_auc_score)
        best_model = model_fit(name=name, model=model, params_grid=params_grid, score=score, X=X_train, y=y_train)
        save_model(name=name, model=best_model)

    with timer('data validate'):
        model_score = model_validate(name=name, model=best_model, score=roc_auc_score, X=X_valid, y=y_valid)
        save_score(name=name, score=model_score)

    with timer('data submit'):
        model_score_predict = model_submit(model=best_model, X=data_test)
        model_prediction = pd.DataFrame(model_score_predict.reshape(-1, 1), columns=['score'])
        model_index = pd.DataFrame(data_all_test['product_id'], columns=['product_id'])
        data_submit = model_index.join(model_prediction)
        save_prediction(name=name, prediction=data_submit[['product_id', 'score']])


if __name__ == "__main__":
    main()
