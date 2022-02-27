API_URL = "http://localhost:8000"

DEBUG = True

TARGET_PATH = {
    "file_metadata": "file/metadata/",
    "task": "task/",
    "experiment": "experiment",
    "predict": "predict",
    "model": "model",
    "evaluation": "evaluation",
}

TASK_STATUS = {
    'PENDING': 0,
    'STARTED': 1,
    'SUCCESS': 2,
    'FAILURE': 3,
    'RETRY': 4,
    'REVOKED': 5,
}

MAX_SAMPLE = 1000

# AUTOGLUON MODEL SETTING
FEATURE_GENERATOR = {
    'enable_text_ngram_features': False,
    'enable_text_special_features': False
}

HYPER_PARAMETER = {
    # 'NN': {},
    'GBM': [
        {},
        {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
        'GBMLarge',
    ],
    'CAT': {},
    'XGB': {},
    # 'FASTAI': {},
    'RF': [
        {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
        {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
        {'criterion': 'mse', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
    ],
    'XT': [
        {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
        {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
        {'criterion': 'mse', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
    ],
    'KNN': [
        {'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}},
        {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}},
    ],
}
