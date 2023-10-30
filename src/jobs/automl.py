from utils import config
from utils.transform import Input, Output, Excel, get_experiments_url, get_predict_url, get_experiments_dataset_url, \
    get_file_url, distribution_density, get_evaluation_url, Histogram
from utils import services
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score, \
    confusion_matrix, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, \
    mean_absolute_percentage_error
import numpy as np
from autogluon.tabular import TabularPredictor as task
import pandas as pd
import shap
from sklearn2pmml import sklearn2pmml
from nyoka import skl_to_pmml, lgb_to_pmml
from sklearn.pipeline import Pipeline
from teradataml.context.context import *
from teradataml import save_byom, delete_byom
from autogluon.features.generators.astype import AsTypeFeatureGenerator
from autogluon.features.generators import CategoryFeatureGenerator
from sklearn.base import BaseEstimator, ClassifierMixin


def parse_float(val):
    try:
        if np.isnan(val):
            return None
        return float(val)
    except:
        return val


class LGBMBoosterWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, booster):
        self.booster = booster

    def fit(self, X, y=None):
        # Just return self as fitting is already done when the booster was trained.
        return self

    def predict(self, X):
        return self.booster.predict(X)

    def predict_proba(self, X):
        return self.booster.predict(X, raw_score=False)

class ModelWrapper:
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def predict(self, data):
        return np.array(self.model.predict(pd.DataFrame(data, columns=self.features)))

    def predict_prob(self, data):
        predict_value = np.array(self.model.predict_proba(pd.DataFrame(data, columns=self.features)))

        if predict_value.ndim == 1:
            predict_value = np.array([predict_value, 1 - predict_value]).T

        return predict_value


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
            train_data=df_train[self.experiment_info.get("features") + [self.experiment_info.get("target")]],
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
        # df_data = df_data[self.experiment_info.get("features") + [self.experiment_info.get("target")]]

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

            _rename = dict(zip(_predict_result.columns, ["predict_class_{}".format(str(label).strip()) for label
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
        self.threshold_num = 100

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

        self.base_metric()

        if self.predictor.problem_type == "regression":
            self.evaluation_predict_actual()
        else:
            self.evaluation_class()

        self.export_evaluation_result()

    def base_metric(self):
        """
        Calculation basic score for both regression and classification
        :return:
        """
        # Evaluation other scores
        scores = self.predictor.evaluate(self.df_data, model=self.evaluation_info.get("model_name"), silent=True,
                                         detailed_report=True)

        scores.pop("confusion_matrix", None)
        scores.pop("classification_report", None)
        # Post evaluation to server
        _data = {
            "evaluation_id": self.evaluation_id,
            "scores": scores,
        }

        services.post(data=_data, target_path=config.TARGET_PATH.get("evaluation"))

    def evaluation_class(self):
        df_y_predict_prob = self.predictor.predict_proba(self.df_data, self.evaluation_info.get("model_name"))
        df_y_actual = self.df_data[self.predictor.label]
        for cur_class in self.predictor.class_labels_internal_map.keys():
            y_actual = df_y_actual.apply(lambda x: 1 if x == cur_class else 0)
            y_predict_prob = df_y_predict_prob[cur_class].values

            class_id = "{}_{}".format(self.evaluation_id, str(cur_class).strip().lower())

            # Post data to save class info
            data = {
                "evaluation_id": self.evaluation_id,
                "class_id": class_id,
                "class_name": str(cur_class).strip().lower()
            }

            services.post(data=data, target_path=config.TARGET_PATH.get("evaluation_class"))

            self.eval_class_roc_lift(y_actual, y_predict_prob, class_id)
            self.eval_class_predict_distri(y_actual, y_predict_prob, class_id)

    def evaluation_predict_actual(self):
        """
        Predict - Actual
        :return:
        """
        if len(self.df_data) > config.MAX_SAMPLE:
            _df_data = self.df_data.sample(n=config.MAX_SAMPLE, replace=True)
        else:
            _df_data = self.df_data

        # Not change location of actual and predict
        predict_vs_actual = {
            "actual": list(_df_data[self.predictor.label].values),
            "predict": self.predictor.predict(_df_data, self.evaluation_info.get("model_name")).values.astype(float).tolist()
        }

        data = {
            "evaluation_id": self.evaluation_id,
            "predict_vs_actual": predict_vs_actual,
        }

        services.post(data=data, target_path=config.TARGET_PATH.get("evaluation_predict_actual"))

    def eval_class_roc_lift(self, y_actual, y_predict_prob, class_id):
        scores = []

        base_value = 0

        for threshold in np.linspace(0, 1, self.threshold_num + 1):
            threshold = round(threshold, 3)

            y_predict = np.where(y_predict_prob > threshold, 1, 0)

            tn, fp, fn, tp = confusion_matrix(y_actual, y_predict).ravel()

            recall = recall_score(y_actual, y_predict)

            precision = precision_score(y_actual, y_predict)

            accuracy = accuracy_score(y_actual, y_predict)

            f1 = f1_score(y_actual, y_predict)

            # False Positive Rate.
            """ Ti le du doan sai tai not obese class.
            Ex : We have 2 class obese(beo phi) and not obese.
            FPR proportion of not obese samples that were incorrectly classified 
            """
            fpr = fp / (fp + tn)

            # True Positive Rate
            """ Ti le du doan dung tai obese class
            Proportion of obese sample that were correctly classified
            """
            tpr = tp / (tp + fn)

            # Positive predictive value
            ppv = tp / (tp + fp)

            # Base value
            base_value = ppv if threshold == 0 else base_value

            """ Overall population
            Ung voi threshold dang xet, co bao nhieu % data dang duoc su dung.
            Threshold = 0. -> 100% data su dung
            Threshold = 10. -> xx% data duoc su dung. Cach tinh xx = Count( data[threshold > 10] )  
            """
            overall_population = np.where(y_predict_prob >= threshold, 1, 0).sum() / len(self.df_data)

            """Target population
            Ti le du doan dung tren toan bo target
            """
            target_population = tp / y_actual.sum()

            top_percent_of_predict = (tp + fp) / (tp + tn + fp + fn)

            _scores = {
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                'fpr': fpr, 'tpr': tpr, 'ppv': ppv, 'recall': recall,
                'f1': f1, 'precision': precision,
                'accuracy': accuracy, 'base_value': base_value,
                'threshold': threshold,
                'top_percent_of_predict': top_percent_of_predict,
                'overall_population': overall_population,
                'target_population': target_population,
            }

            _scores = dict([key, parse_float(value)] for key, value in _scores.items())
            scores.append(_scores)

        data = {
            "class_id": class_id,
            "scores": scores
        }

        services.post(data=data, target_path=config.TARGET_PATH.get("evaluation_class_roc_lift"))

    def eval_class_predict_distri(self, y_actual, y_predict_prob, class_id):
        """
        Distribution probability of distribution
        :return:
        """
        df_predict_distribution = pd.DataFrame({"predict_prob":  np.linspace(0, 1, self.threshold_num + 1)})

        df = pd.DataFrame({"predict_prob":  np.round(y_predict_prob, 2), "y_true": y_actual})

        df_cur_class, df_left_class = df.loc[df.y_true == 1], df.loc[df.y_true == 0]

        _target_class = distribution_density(df_cur_class.predict_prob.values, self.threshold_num + 1)
        _left_class = distribution_density(df_left_class.predict_prob.values, self.threshold_num + 1)

        df_predict_distribution["target_class_density"] = _target_class
        df_predict_distribution["left_class_density"] = _left_class

        # Post predict distribution to API Server

        data = {
            "class_id": class_id,
            "predict_distribution": df_predict_distribution.to_dict("list")
        }
        services.post(data=data, target_path=config.TARGET_PATH.get("evaluation_class_distribution"))

    def export_evaluation_result(self):
        _df_predict = self.df_data.copy()

        # Predict
        if self.predictor.problem_type == "regression":
            _predict_result = self.predictor.predict(_df_predict, self.evaluation_info.get("model_name"))

            _df_predict["Predict"] = _predict_result
        else:
            _predict_result = self.predictor.predict_proba(_df_predict, self.evaluation_info.get("model_name"))

            _rename = dict(zip(_predict_result.columns, ["predict_class_{}".format(str(label).strip()) for label
                                                         in _predict_result.columns.tolist()]))

            _predict_result.rename(columns=_rename, inplace=True)

            _df_predict = pd.concat([_df_predict, _predict_result], axis=1)

        # Save file into local
        file_path = get_evaluation_url(self.evaluation_id)
        try:
            excel = Excel(file_path)

            excel.add_data(worksheet_name="Data", pd_data=_df_predict, header_lv0=None, is_fill_color_scale=False,
                           columns_order=None)

            excel.save()
        except Exception as e:
            mes = 'Can not generate excel file.  ERROR : {}'.format(e)
            raise Exception(mes)

    def sub_population(self, sub_population_id, column_name):
        """
        Call this function by run special task. Replace data from DB every time when it called
        :return:
        """

        # TODO : Input which value is positive.
        actual = self.predictor.transform_labels(self.df_data[self.predictor.label].values).values
        predict_raw = self.predictor.predict(self.df_data, self.evaluation_info.get("model_name"))
        predict = self.predictor.transform_labels(predict_raw).values
        feature = self.df_data[column_name].values

        df_sub = pd.DataFrame(
            {
                "feature_name": column_name,
                "feature_value": feature,
                "predict": predict,
                "actual": actual,
                "actual_raw": self.df_data[self.predictor.label].values,
                "predict_raw": predict_raw
            })

        hist = Histogram(df_sub, "feature_value")
        df_allocated_group = hist.pd_data

        if self.predictor.problem_type == "binary":
            df_prob = self.predictor.predict_proba(self.df_data, self.evaluation_info.get("model_name"))
            df_prob.reset_index(drop=True, inplace=True)
            df_allocated_group = pd.concat([df_allocated_group, df_prob], axis=1)

        if self.predictor.problem_type == "regression":
            df_sub_population = self._regression_score(df_allocated_group)
        else:
            # Calculation subpopulation for each class
            df_sub_population = self._binary_score(df_allocated_group, self.predictor.problem_type,
                                                   list(self.predictor.class_labels_internal_map.keys()))

        data = {
            "sub_population_id": sub_population_id,
            "sub_population": df_sub_population.to_dict("records")
        }
        services.post(data, target_path=config.TARGET_PATH.get("evaluation_sub_population"))

    @staticmethod
    def _regression_score(pd_data):
        df_grouped = pd.DataFrame()

        for group in pd_data.group_name.unique():
            df_group = pd_data.loc[pd_data.group_name == group]

            _df_grouped = df_group.groupby(['group_order', 'group_name', 'is_outlier']).agg({"actual": "count"}).rename(
                columns={"actual": "sample"})

            _df_grouped["mean_absolute_error"] = mean_absolute_error(df_group.actual.values, df_group.predict.values)
            _df_grouped["mean_squared_error"] = mean_squared_error(df_group.actual.values, df_group.predict.values)
            _df_grouped["median_absolute_error"] = median_absolute_error(df_group.actual.values,
                                                                         df_group.predict.values)
            _df_grouped["r2"] = r2_score(df_group.actual.values, df_group.predict.values)
            _df_grouped["mean_absolute_percentage_error"] = mean_absolute_percentage_error(df_group.actual.values,
                                                                                           df_group.predict.values)

            df_grouped = pd.concat([df_grouped, _df_grouped])

        df_grouped["sample_percent"] = df_grouped["sample"] / len(pd_data)
        df_grouped = df_grouped.replace([np.nan], [None])

        df_grouped.sort_values("group_order", inplace=True)
        df_grouped.reset_index(inplace=True)

        return df_grouped.reset_index()

    @staticmethod
    def _binary_score(df_data, problem_type, class_list):
        """

        :param pd_data: Allocated group data
        :param problem_type: binary or multiple
        :param class_list: List of target class.
        :return:
        """
        df_grouped = pd.DataFrame()

        """
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        - 'micro':
        Calculate metrics globally by counting the total true positives, false negatives and false positives.
        'binary':
        Only report results for the class specified by pos_label. This is applicable only if targets (y_{true,pred}) are binary.
        """

        f1_score_average = "binary" if problem_type == "binary" else "micro"

        if df_data.group_name.nunique() > 100:
            group_filter = np.random.choice(df_data.group_name.unique(), 100, replace=False)
            df_data = df_data.loc[df_data.group_name.isin(group_filter)].reset_index(drop=True)

        for group in df_data.group_name.unique():
            df_group = df_data.loc[df_data.group_name == group]

            _df_grouped = df_group.groupby(['group_order', 'group_name', 'is_outlier']).agg({"actual": "count"}).rename(
                columns={"actual": "sample"})

            _df_grouped["accuracy"] = accuracy_score(df_group.actual.values, df_group.predict.values)
            _df_grouped["f1"] = f1_score(df_group.actual.values, df_group.predict.values,
                                         average=f1_score_average)
            _df_grouped["balanced_accuracy_score"] = balanced_accuracy_score(df_group.actual.values,
                                                                             df_group.predict.values)
            _df_grouped["precision_score"] = precision_score(df_group.actual.values, df_group.predict.values,
                                                             average=f1_score_average)
            _df_grouped["recall_score"] = recall_score(df_group.actual.values, df_group.predict.values,
                                                       average=f1_score_average)

            # Calculation confusion matrix
            labels = df_group["actual_raw"].unique().tolist()
            matrix = confusion_matrix(df_group.actual_raw.values, df_group.predict_raw.values, labels).tolist()
            confusion = [{
                "labels": labels,
                "matrix": matrix
            }]
            _df_grouped["confusion"] = confusion

            # TODO: Create distribution chart for binary
            if problem_type == "binary":
                density = []
                for target_class in class_list:
                    cur_class_prob = df_group.loc[df_group.actual_raw == target_class][target_class].values

                    if len(cur_class_prob) == 0:
                        dens, prob = [], []
                    else:
                        dens = distribution_density(cur_class_prob, 30)
                        prob = np.round(np.linspace(0, 1, 30), 2)

                    _dens = {
                        "series": [{"x": x, "y": y} for x, y in zip(prob, dens)],
                        "name": target_class
                    }
                    density.append(_dens)

                _df_grouped["density"] = [density]

            df_grouped = pd.concat([df_grouped, _df_grouped])

        df_grouped["sample_percent"] = df_grouped["sample"] / len(df_data)

        df_grouped = df_grouped.replace([np.nan], [None])

        # Order group by group order
        df_grouped.sort_values("group_order", inplace=True)
        df_grouped.reset_index(inplace=True)

        return df_grouped


class Explain:
    """
    Used SHAP to interpretation models
    """
    def __init__(self, explain_id):

        self.explain_id = explain_id

        # Load explain info
        params = {"explain_id": explain_id}
        self.explain_info = services.get(target_path=config.TARGET_PATH.get("explain"), params=params)

        if self.explain_info.get("file_id", None) is None:
            # Load test data
            test_url = get_experiments_dataset_url(self.explain_info.get("experiment_id"), "test.pickle")

            self.df_data = Input(test_url).from_pickle()

            self.df_data = self.df_data.sample(100) if len(self.df_data) > 100 else self.df_data

        # Load predictor
        _save_dir = get_experiments_url(self.explain_info.get("experiment_id"))
        self.predictor = task.load(_save_dir)

        # Calculation shap
        self.shap_values = self.shap_calculation()

    def explain(self):

        self.pdp()

    def pdp(self):

        pdp_list = []

        for feature in self.predictor.features():

            if self.predictor.problem_type == "regression":
                pdp_values = self.pdp_regress(feature)
            else:
                # Create
                pdp_values = self.pdp_class(feature)

            pdp_list.append(
                {
                    "feature": feature,
                    "pdp_values": pdp_values
                }
            )

        data = {
            "explain_id": self.explain_id,
            "pdp_list": pdp_list
        }
        services.post(data=data, target_path=config.TARGET_PATH.get("explain_pdp"))

    def pdp_regress(self, feature):

        df_shap = self.shap_values

        expected_value = df_shap["expected_value"].unique()[0]

        df_pdp = pd.DataFrame(
            {
                "feature_name": feature,
                "feature_value": self.df_data[feature].values,
                "shap": df_shap[feature].values,
            })

        df_pdp["shap"] = df_pdp["shap"] + expected_value

        hist = Histogram(df_pdp, "feature_value")
        df_pdp_grouped = hist.pd_data.groupby(['group_order', 'group_name']).agg(pdp_value=('shap', "mean"),
                                                                                 num=('shap', "count"))
        df_pdp_grouped.reset_index(inplace=True)

        return df_pdp_grouped.to_dict("records")

    def pdp_class(self, feature):

        pdp_class = []

        for class_decode in self.predictor.class_labels_internal_map.keys():

            df_shap = self.shap_values[class_decode]

            expected_value = df_shap["expected_value"].unique()[0]

            df_pdp = pd.DataFrame(
                {
                    "feature_name": feature,
                    "feature_value": self.df_data[feature].values,
                    "shap": df_shap[feature].values,
                })

            df_pdp["shap"] = df_pdp["shap"] + expected_value

            hist = Histogram(df_pdp, "feature_value")

            df_pdp_grouped = hist.pd_data.groupby(['group_order', 'group_name']).agg(pdp_value=('shap', "mean"),
                                                                                     num=('shap', "count"))

            df_pdp_grouped.reset_index(inplace=True)

            pdp_class.append({
                "class_name": str(class_decode).strip(),
                "pdp_values": df_pdp_grouped.to_dict("records")
            })

        return pdp_class

    def shap_calculation(self):

        # Calculation shap
        model_name = self.explain_info.get("model_name")
        try:
            model = self.predictor._trainer.load_model(model_name)
        except Exception as e:
            mes = "Can't load model {}. ERROR : {}".format(model_name, e)
            raise mes
        model_type = model.__class__.__name__

        is_tree_explain = model_type in ['RFModel', 'XTModel']

        if is_tree_explain:
            df_data_trans = self.predictor.transform_features(data=self.df_data, model=model_name)

            data = model.preprocess(df_data_trans)
            explainer = shap.TreeExplainer(model.model)
            shap_values = explainer.shap_values(data)

        else:
            # Load and Transform train data
            train_url = get_experiments_dataset_url(self.explain_info.get("experiment_id"), "train.pickle")
            df_train = Input(train_url).from_pickle()

            df_train_transform = self.predictor.transform_features(df_train)
            train_summary = shap.kmeans(df_train_transform, 150).data

            # 3. Create model wrapper
            model_wrapper = ModelWrapper(model, self.predictor.features())

            # 4. Shap calculation
            if self.predictor.problem_type != "regression":
                explainer = shap.KernelExplainer(model_wrapper.predict_prob, train_summary)
            else:
                explainer = shap.KernelExplainer(model_wrapper.predict, train_summary)

            # 5. Shap calculation for input data
            df_data_trans = self.predictor.transform_features(data=self.df_data)
            shap_values = explainer.shap_values(df_data_trans.values)

        if self.predictor.problem_type == "regression":
            df_shap = pd.DataFrame(shap_values, columns=model.features)
            df_shap["expected_value"] = explainer.expected_value
            return df_shap
        else:
            shap_values_multi_class = {}
            for class_decode, class_encode in self.predictor.class_labels_internal_map.items():
                df_shap = pd.DataFrame(shap_values[class_encode], columns=model.features)
                df_shap["expected_value"] = explainer.expected_value[class_encode]

                shap_values_multi_class[class_decode] = df_shap

            return shap_values_multi_class


class Deployment:
    def __init__(self, deploy_id):
        self.deploy_id = deploy_id

        # Load explain info
        params = {"deploy_id": deploy_id}
        _info = services.get(target_path=config.TARGET_PATH.get("deploy"), params=params)

        self.deploy_info = _info.get("deploy_info")
        self.connection_info = _info.get("connection_info")

        # Load predictor
        self._save_dir = get_experiments_url(self.deploy_info.get("experiment_id"))

        self.predictor, self.model = self._get_predictor()

    def deploy(self):
        self.export_sklearn_model(self.deploy_info.get("model_name"))

        sql_preprocess = self.generate_preprocessing_sql()
        sql_prediction = self.generate_prediction_sql()

        # Save this sql back to DB
        _data = {
            "deploy_id": self.deploy_info.get("deploy_id"),
            "sql_preprocess": sql_preprocess,
            "sql_prediction": sql_prediction
        }
        services.post(data=_data, target_path=config.TARGET_PATH.get("deploy"))

    def _get_predictor(self):
        predictor = task.load(self._save_dir)
        model = predictor._trainer.load_model(self.deploy_info.get("model_name"))

        return predictor, model

    def export_sklearn_model(self, model_name):
        # Connection to DB
        create_context(host=self.connection_info.get("host_name"), username=self.connection_info.get("username"),
                       password=self.connection_info.get("password"), database=self.connection_info.get("database"))

        # Load model
        model = self.predictor._trainer.load_model(model_name)
        if model.__class__.__name__ not in ['RFModel', 'KNNModel', 'XTModel', 'LGBModel']:
            raise Exception("Only support for Random Forest and KNN Model")

        file_name = "{}.pmml".format(self.deploy_info.get("teradata_model_id"))
        file_path = os.path.join(self._save_dir, file_name)

        # Save model into local
        if model.__class__.__name__ in ['RFModel', 'XTModel']:
            sklearn2pmml(model.model, file_path)

        elif model.__class__.__name__ == 'KNNModel':
            pipeline = Pipeline([("model", model.model)])

            # Export the pipeline to PMML
            features = ["x{}".format(i+1) for i in range(len(model.features))]
            skl_to_pmml(pipeline, features, "target", file_path)

        elif model.__class__.__name__ == 'LGBModel':
            wrapped_booster = LGBMBoosterWrapper(model.model)
            pipeline = Pipeline([("model", wrapped_booster)])
            features = ["x{}".format(i+1) for i in range(len(model.features))]
            lgb_to_pmml(pipeline, features, "target", file_path)

        # Save model into Teradata
        try:
            # Saving model in PMML format to Vantage table
            save_byom(model_id=self.deploy_info.get("teradata_model_id"), model_file=file_path,
                      table_name=self.deploy_info.get("output_table"))

        except Exception as e:
            if str(e.args).find('TDML_2200') >= 1:
                delete_byom(model_id=model_name, table_name=self.deploy_info.get("output_table"))
                save_byom(model_id=model_name, model_file=file_path, table_name=self.deploy_info.get("output_table"))

    def _get_generator(self):
        """
        Extract generator form predictor
        """
        f = self.predictor._learner.feature_generators[0]

        cat_f_generator, as_type_f_generator = None, None
        label_encode_cols = []

        for generators in f.generators:
            for generator in generators:
                if isinstance(generator, AsTypeFeatureGenerator):
                    as_type_f_generator = generator
                if isinstance(generator, CategoryFeatureGenerator):
                    cat_f_generator = generator
                    label_encode_cols = cat_f_generator.category_map.keys()

        return as_type_f_generator, cat_f_generator, label_encode_cols

    def generate_preprocessing_sql(self):

        as_type_f_generator, cat_f_generator, label_encode_cols = self._get_generator()

        ############ 1. DATA TYPE CONVERT ##############
        # FOR NOT DUPLICATE
        processed_column = []

        # 1. Boolean feature.
        bool_features = as_type_f_generator._bool_features

        sql_dtype_bool = ""
        for col, val in bool_features.items():
            if (col not in label_encode_cols) and (col not in processed_column):
                sql_dtype_bool += "\t CASE WHEN \"{}\" = '{}' THEN 1 ELSE 0 END AS \"{}\", \r\n".format(col, val, col)
                processed_column.append(col)

        # 2. Number Feature.Fill null value with 0
        sql_dtype_numeric = ""
        for col in as_type_f_generator._int_features:
            if (col not in label_encode_cols) and (col not in processed_column):
                sql_dtype_numeric += "\t COALESCE(CAST(\"{}\" AS INT), 0) AS \"{}\", \r\n".format(col, col)
                processed_column.append(col)

        # 3. Category feature.
        sql_dtype_category = ""
        if hasattr(self.model, "_feature_generator"):
            for cat_fea in self.model._feature_generator.features_in:
                if (cat_fea not in label_encode_cols) and (cat_fea not in processed_column):
                    sql_dtype_category += "\t COALESCE(CAST(\"{}\" AS VARCHAR(255)), '') AS \"{}\", \r\n".format(cat_fea,
                                                                                                         cat_fea)
                    processed_column.append(cat_fea)

        # 4. Float features
        sql_dtype_float = ""
        float_features = [key for key, value in as_type_f_generator._type_map_real_opt.items() if
                          value == np.dtype('float64')]
        for col in float_features:
            sql_dtype_float += "\t COALESCE(CAST(\"{}\" AS FLOAT), 0) AS \"{}\", \r\n".format(col, col)

        ######## FINISH DATA PROCESSING ###########

        ######## 2. LABEL ENCODING PROCESSING ########
        # Label encoding
        sql_encoding = ""
        if cat_f_generator is not None:
            for col, encode_values in cat_f_generator.category_map.items():
                _sql_when_codition = "\t CASE \r\n"

                for i, encode_val in enumerate(encode_values):
                    _sql_when_codition += "\t\t WHEN \"{}\" = '{}' THEN {} \r\n".format(col, encode_val, i)

                _sql_when_codition += "\tELSE -1 \r\n"
                _sql_when_codition += "\tEND AS \"{}\", \r\n".format(col)

                sql_encoding += _sql_when_codition

        ######## FINISH LABEL ENCODING PROCESSING ###########

        # Remove last comma
        sql_encoding = sql_encoding[:sql_encoding.rfind(",")]
        features_order = ", ".join(f"\"{col}\" as x{i + 1}" for i, col in enumerate(self.model.features))
        sql_uid = "ROW_NUMBER() OVER (ORDER BY \"{}\") AS id,".format(self.model.features[0])

        return "CREATE VIEW [OUTPUT_VIEW] as \r\n" + \
            " SELECT " + sql_uid + features_order + "\r\nFROM (SELECT \r\n" + \
            sql_dtype_bool + sql_dtype_numeric + sql_dtype_float + sql_dtype_category + sql_encoding + \
            "\r\nFROM " + "[INTPUT_TABLE]" + ") predict_table;"

    def generate_prediction_sql(self):
        # sql_prediction
        sql = "SELECT * FROM mldb.PMMLPredict( \r\n" + \
                "\tON [OUTPUT_VIEW] as InputTable\r\n" + \
                "\tON \r\n" + \
                "\t(SELECT * \r\n" + \
                "\tFROM {}.{} as ModelTable \r\n".format(self.connection_info.get("database"), self.deploy_info.get("output_table")) + \
                "\tWHERE model_id = '{}')\r\n".format(self.deploy_info.get("teradata_model_id")) + \
                "DIMENSION USING Accumulate ( 'id' )) as T;"

        return sql

