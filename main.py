from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from hyperparam_tuning import tune_with_grid_search, tune_with_halving_grid_search

RANDOM_STATE = 35090


def main():
    # use fixed random state for repeatable data set
    X, Y = make_classification(n_samples=3000, random_state=RANDOM_STATE)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=RANDOM_STATE)

    svc_params = {
        'C': [0.1, 0.5, 1, 2, 5, 10],
        'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
        'tol': [1e-3, 1e-2]
    }

    gs_results, gs_duration = tune_with_grid_search(x_train, y_train, svc_params)
    halving_results, halving_duration = tune_with_halving_grid_search(x_train, y_train, svc_params)

    print(gs_results.head())
    print(halving_results.head())

    score1 = gs_results['mean_test_score'].iloc[0]
    params1 = gs_results['params'].iloc[0]
    score2 = halving_results['mean_test_score'].iloc[0]
    params2 = halving_results['params'].iloc[0]

    svc1 = svm.SVC(**params1)
    svc1.fit(x_train, y_train)
    accuracy1 = accuracy_score(y_test, svc1.predict(x_test))

    svc2 = svm.SVC(**params2)
    svc2.fit(x_train, y_train)
    accuracy2 = accuracy_score(y_test, svc2.predict(x_test))

    print(f'Best score for GridSearchCv is {score1:.3f}, took {gs_duration:.2f} seconds')
    print(f'Params: {params1}')
    print(f'Corresponding test accuracy: {accuracy1 * 100:.2f}%\n')

    print(f'Best score for HalvingGridSearchCv is {score2:.3f}, took {halving_duration:.2f} seconds')
    print(f'Params: {params2}')
    print(f'Corresponding test accuracy: {accuracy2 * 100:.2f}%')


if __name__ == '__main__':
    main()
