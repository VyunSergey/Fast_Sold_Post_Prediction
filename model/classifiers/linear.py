from sklearn.linear_model import SGDClassifier

linear_name = 'SGDClassifier'

linear_params_grid = {
    'alpha': [0.0001],
    'loss': ['log'],
    #'loss': ['hinge', 'log'],
    'max_iter': [1000],
    'tol': [0.001, 0.0001, 0.01, 0.1]
}

linear = SGDClassifier(random_state=42)

if __name__ == "__main__":
    print(linear.get_params())
