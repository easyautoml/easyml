from utils import config
from utils.transform import Input, Output, Excel, get_experiments_url, get_predict_url, get_experiments_dataset_url, get_file_url
from utils import services
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, accuracy_score
import numpy as np
from autogluon.tabular import TabularPredictor as task
import pandas as pd


class Train:
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id

        params = {"experiment_id": experiment_id}
        self.experiment_info = services.get(target_path=config.TARGET_PATH.get("experiment"), params=params)

        # Model setting
        self.preset = 'medium_quality_faster_train'
        self.time_limit = 4200
        self.feature_generator = config.FEATURE_GENERATOR
        self.hyper = config.HYPER_PARAMETER

    def train(self):
        # Load data
        file_url = get_file_url(self.experiment_info.get("file_id"))
        df_data = Input(file_url).from_csv()

        if config.DEBUG:
            df_data = df_data.head(100)

        # Split train test
        df_train, df_test = self.pre_process(df_data)

        # Build model
        output_path = get_experiments_url(self.experiment_id)

        predictor = task(label=self.experiment_info.get("target"),
                         path=output_path,
                         problem_type=self.experiment_info.get("problem_type"),
                         eval_metric=self.experiment_info.get("score"),
                         verbosity=1).fit(
            train_data=df_train,
            presets=self.preset,
            time_limit=self.time_limit,
            hyperparameters=self.hyper,
            _feature_generator_kwargs=self.feature_generator,
        )

        # Get models info
        self.models(predictor, df_test)

        # Update Best model
        _data = {
            "experiment_id": self.experiment_id,
            "best_model_id": self._get_model_id(predictor.get_model_best())
        }
        services.post(data=_data, target_path=config.TARGET_PATH.get("experiment"))

        # Save test data into experiments folder
        train_url = get_experiments_dataset_url(self.experiment_id, "train.pickle")
        test_url = get_experiments_dataset_url(self.experiment_id, "test.pickle")

        Output(train_url).to_pickle(df_train)
        Output(test_url).to_pickle(df_test)

    def pre_process(self, df_data):
        # Drop Null at target
        df_data.dropna(subset=[self.experiment_info.get("target")], inplace=True)

        # Fill Null by 0
        df_data.fillna(0, inplace=True)

        # Keep only features column and target
        df_data = df_data[self.experiment_info.get("features") + [self.experiment_info.get("target")]]

        # Split train
        df_train = df_data.sample(frac=float(self.experiment_info.get("split_ratio")) * 0.1)
        df_test = df_data.drop(df_train.index)

        return df_train, df_test

    def models(self, predictor, df_test):
        df_models_info = predictor.leaderboard(df_test, silent=True)

        _rename = {"pred_time_test": "predict_time", "model": "model_name"}
        df_models_info.rename(columns=_rename, inplace=True)

        # Importance features calculation
        for model_name in predictor.get_model_names():
            _f_impt = predictor.feature_importance(data=df_test, model=model_name, feature_stage='original',
                                                   num_shuffle_sets=1,
                                                   include_confidence_band=False)["importance"].astype(np.float16)

            df_models_info.loc[df_models_info.model_name == model_name, "features_importance"] = [_f_impt.to_dict()]

        df_models_info[["score_test", "score_val"]] = df_models_info[["score_test", "score_val"]].abs()
        df_models_info["model_id"] = df_models_info["model_name"].apply(lambda x: self._get_model_id(x))

        _info = ["model_id", "model_name", "score_test", "score_val", "fit_time", "predict_time", "features_importance"]
        models_list = df_models_info[_info].to_dict("records")

        # Post model list to server
        _data = {
            "experiment_id": self.experiment_id,
            "models_list": models_list
        }
        services.post(data=_data, target_path=config.TARGET_PATH.get("model"))

    def _get_model_id(self, model_name):
        return "{}_{}".format(self.experiment_id, model_name)


class Predict:
    def __init__(self, predict_id):
        self.predict_id = predict_id

        params = {"predict_id": self.predict_id}
        self.predict_info = services.get(config.TARGET_PATH.get("predict"), params=params)
        self.experiment_info = self.predict_info.get("experiment")

    def predict(self):
        # Load experiment id
        _save_dir = get_experiments_url(self.experiment_info.get("experiment_id"))
        predictor = task.load(_save_dir)

        # Load data
        file_url = get_file_url(self.predict_info.get("file_id"))
        df_predict = Input(file_url).from_csv()

        # Predict
        if self.experiment_info.get("problem_type") == "regression":
            _predict_result = predictor.predict(df_predict, self.predict_info.get("model_name"))

            df_predict["Predict"] = _predict_result
        else:
            _predict_result = predictor.predict_proba(df_predict, self.predict_info.get("model_name"))

            _rename = dict(zip(_predict_result.columns, ["predict_class_{}".format(label.strip()) for label
                                                         in _predict_result.columns.tolist()]))

            _predict_result.rename(columns=_rename, inplace=True)

            df_predict = pd.concat([df_predict, _predict_result], axis=1)

        # Save file into local
        file_path = get_predict_url(self.predict_id)
        try:
            excel = Excel(file_path)

            excel.add_data(worksheet_name="Data", pd_data=df_predict, header_lv0=None, is_fill_color_scale=False,
                           columns_order=None)

            excel.save()
        except Exception as e:
            mes = 'Can not generate excel file.  ERROR : {}'.format(e)
            raise Exception(mes)


class Evaluation:

    def __init__(self, evaluation_id):
        self.evaluation_id = evaluation_id

        params = {"evaluation_id": self.evaluation_id}
        self.evaluation_info = services.get(config.TARGET_PATH.get("evaluation"), params=params)

        # Step 1. Load predictor
        _save_dir = get_experiments_url(self.evaluation_info.get("experiment_id"))
        self.predictor = task.load(_save_dir)

        # Step 2. Load data
        if self.evaluation_info.get("file_id", None) is None:
            # Load test data
            test_url = get_experiments_dataset_url(self.evaluation_info.get("experiment_id"), "test.pickle")

            self.df_data = Input(test_url).from_pickle()


    def evaluate(self):
        """
        Calculation basic score.
        :return:
        """
        # Evaluation other scores
        scores = self.predictor.evaluate(self.df_data, model=self.evaluation_info.get("model_name"), silent=True,
                                    detailed_report=True)

        # Confusion matrix
        _confusion_matrix, predict_vs_actual = None, None
        if self.predictor.problem_type == "regression":
            if len(self.df_data) > config.MAX_SAMPLE:
                _df_data = self.df_data.sample(n=config.MAX_SAMPLE, replace=True)

            predict_val = self.predictor.predict(_df_data, self.evaluation_info.get("model_name")).values
            actual_val = _df_data[self.predictor.label].values

            # Not change location of actual and predict
            predict_vs_actual = np.array([actual_val, predict_val]).T.tolist()
        else:
            _confusion_matrix = scores.pop("confusion_matrix").to_dict()
            scores.pop("classification_report")

        # Post evaluation to server
        _data = {
            "evaluation_id": self.evaluation_id,
            "scores": scores,
            "confusion_matrix": _confusion_matrix,
            "predict_vs_actual": predict_vs_actual
        }
        services.post(data=_data, target_path=config.TARGET_PATH.get("evaluation"))

    def roc_lift_chart(self):
        """
        Calculation roc and lift chart. This function only use for classification problem
        :return:
        """
        threshold_num = 100

        y_predict_prob = self.predictor.predict_prob(self.df_data, self.evaluation_info.get("model_name")).values
        y_actual = self.df_data[self.predictor.label].values

        for cur_class in self.predictor.class_labels_internal_map.keys():

            roc_scores = []

            base_value = 0

            for threshold in np.linspace(0, 1, threshold_num):
                threshold = round(threshold, 3)

                y_predict = np.where(y_predict_prob >= threshold, 1, 0)

                tn, fp, fn, tp = confusion_matrix(y_actual, y_predict).ravel()

                recall = recall_score(y_actual, y_predict)

                precision = precision_score(y_actual, y_predict)

                accuracy = accuracy_score(y_actual, y_predict)

                f1 = f1_score(y_actual, y_predict)

                # False Positive Rate
                fpr = fp / (fp + tn)

                # True Positive Rate
                tpr = tp / (tp + fn)

                # Positive predictive value
                ppv = tp / (tp + fp)

                # Base value
                base_value = ppv if threshold == 0 else base_value

                top_percent_of_predict = (tp + fp) / (tp + tn + fp + fn)

                roc_scores.append({
                    'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                    'fpr': fpr, 'tpr': tpr, 'ppv': ppv, 'recall': recall,
                    'f1': f1, 'precision': precision,
                    'accuracy': accuracy, 'base_value': base_value,
                    'threshold': threshold,
                    'top_percent_of_predict': top_percent_of_predict,
                    'class': cur_class
                })

            data = {
                "class_id": ""
            }
            # TODO : Post this data to API
            services.post()
