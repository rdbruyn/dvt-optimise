from sklearn import svm
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.base import clone

from pathlib import Path
from pandas import DataFrame
from time import time

RANDOM_STATE = 35090


def main():
    result_dir = Path.cwd() / 'results'
    if not result_dir.exists():
        result_dir.mkdir()

    X, Y = make_classification(n_samples=3000, random_state=RANDOM_STATE)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=RANDOM_STATE)

    svc_params = {
        'C': [0.1, 0.5, 1, 2, 5, 10],
        'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
        'tol': [1e-3, 1e-2]
    }

    svc = svm.SVC()

    gs_start = time()
    gs_results = GridSearchCV(svc, svc_params, cv=5).fit(x_train, y_train)
    duration = time() - gs_start
    print(f'Optimisation done in {duration:.2f} seconds')

    results = DataFrame(gs_results.cv_results_)
    results.loc[:, 'mean_test_score'] *= 100
    results = results.loc[:, ('rank_test_score', 'mean_test_score', 'params')]
    results.sort_values(by='rank_test_score', ascending=True, inplace=True)
    print(results.head())
    results.to_csv(result_dir / 'svc_results.csv')

    # man
    svc = clone(svc)

    hgs_start = time()
    halving_gs_results = HalvingGridSearchCV(svc, svc_params, cv=5).fit(x_train, y_train)
    hgs_duration = time() - hgs_start
    print(f'Halving optimisation done in {hgs_duration:.2f} seconds')

    halving_results = DataFrame(halving_gs_results.cv_results_)
    halving_results = halving_results.loc[:, ('iter', 'rank_test_score', 'mean_test_score', 'params')]
    halving_results.loc[:, 'mean_test_score'] *= 100
    halving_results.sort_values(by=['iter', 'rank_test_score'], ascending=[False, True], inplace=True)
    print(halving_results.head())
    halving_results.to_csv(result_dir / 'halving_svc_results.csv')

    score1 = results['mean_test_score'].iloc[0]
    params1 = results['params'].iloc[0]
    score2 = halving_results['mean_test_score'].iloc[0]
    params2 = halving_results['params'].iloc[0]

    svc1 = svm.SVC(**params1)
    svc1.fit(x_train, y_train)
    accuracy1 = accuracy_score(y_test, svc1.predict(x_test))

    svc2 = svm.SVC(**params2)
    svc2.fit(x_train, y_train)
    accuracy2 = accuracy_score(y_test, svc2.predict(x_test))

    print(f'Best score for GridSearchCv is        {score1:.5f}, took {duration:.2f} seconds')
    print(f'Params: {params1}')
    print(f'Corresponding test accuracy: {accuracy1 * 100:.2f}%\n')

    print(f'Best score for HalvingGridSearchCv is {score2:.5f}, took {hgs_duration:.2f} seconds')
    print(f'Params: {params2}')
    print(f'Corresponding test accuracy: {accuracy2 * 100:.2f}%')


if __name__ == '__main__':
    main()
