from sklearn import svm
from time import time
from pandas import DataFrame
from pathlib import Path

from sklearn.model_selection import GridSearchCV

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

# make sure that result directory exists before running any of the functions
result_dir = Path.cwd() / 'results'
if not result_dir.exists():
    result_dir.mkdir()


def tune_with_grid_search(x_train, y_train, param_grid):
    svc = svm.SVC()

    start = time()
    gs_results = GridSearchCV(svc, param_grid, cv=5).fit(x_train, y_train)
    duration = time() - start

    results = DataFrame(gs_results.cv_results_)
    results.loc[:, 'mean_test_score'] *= 100
    results.to_csv(result_dir / 'svc_results.csv')

    # take the most relevant columns and sort (for readability)
    results = results.loc[:, ('rank_test_score', 'mean_test_score', 'params')]
    results.sort_values(by='rank_test_score', ascending=True, inplace=True)

    return results, duration


def tune_with_halving_grid_search(x_train, y_train, param_grid):
    svc = svm.SVC()

    start = time()
    halving_gs_results = HalvingGridSearchCV(
        svc,
        param_grid,
        cv=5,
        factor=3,
        min_resources='exhaust'
    ).fit(x_train, y_train)

    duration = time() - start

    results = DataFrame(halving_gs_results.cv_results_)
    results.loc[:, 'mean_test_score'] *= 100
    results.to_csv(result_dir / 'halving_svc_results.csv')

    # take the most relevant columns and sort (for readability). Remember to sort on the iter columns first, so we see
    # the models with the most training data behind them first.
    results = results.loc[:, ('iter', 'rank_test_score', 'mean_test_score', 'params')]
    results.sort_values(by=['iter', 'rank_test_score'], ascending=[False, True], inplace=True)

    return results, duration
