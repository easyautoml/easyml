API_URL = "http://localhost:8000"

DEBUG = False

TARGET_PATH = {
    "file_metadata": "file/metadata/",
    "task": "task/",
    "experiment": "experiment",
    "predict": "predict",
    "model": "model",

    "evaluation": "evaluation",
    "evaluation_sub_population": "evaluation/sub_population",
    "evaluation_predict_actual": "evaluation/evaluation_predict_actual",
    "evaluation_class": "evaluation/evaluation_class",
    "evaluation_class_roc_lift": "evaluation/evaluation_class/roc_lift",
    "evaluation_class_distribution": "evaluation/evaluation_class/distribution",

    "explain": 'explain',
    "explain_pdp": 'explain/pdp',
    "explain_pdp_regress": 'explain/pdp/regress',
    "explain_pdp_class": 'explain/pdp/class',
    "explain_pdp_class_values": 'explain/pdp/class/values',
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
