from sklearn.ensemble import GradientBoostingClassifier

gbc_name = 'GradientBoostingClassifier'

gbc_params_grid = {
    'learning_rate': [0.01],
    # , 0.1, 0.05, 0.5],
    'loss': ['deviance'],
    # , 'exponential'],
    'max_depth': [3, 5],
    # , 10],
    'max_features': [None, 'auto'],
    # , 'sqrt', 'log2'],
    'n_estimators': [200]
    # , 10, 50, 100]
}

gbc = GradientBoostingClassifier(random_state=42)

if __name__ == "__main__":
    print(gbc.get_params())
