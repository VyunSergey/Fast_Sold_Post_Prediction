from sklearn.tree import DecisionTreeClassifier

tree_name = 'DecisionTreeClassifier'

tree_params_grid = {
    'max_depth': [None, 10, 100],
    'max_features': [None, 1000, 100000],
    'max_leaf_nodes': [None, 3, 5],
    'min_samples_split': [2, 3]
}

tree = DecisionTreeClassifier(random_state=42)

if __name__ == "__main__":
    print(tree.get_params())
