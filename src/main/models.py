from django.db import models
from django.apps import apps
from django.conf import settings
from collections import defaultdict
from cryptography.fernet import Fernet
import teradatasql
import base64


SCORE = {
    "root_mean_squared_error": "RMSE",
    "mean_absolute_error": "MAE",
    "r2": "R2",
    "pearsonr": "PEARSON",
    "mean_squared_error": "MSE",
    "mean_absolute_error": "MAE",
    "f1": "F1",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "roc_auc": "ROC AUC",
    "mcc": "MCC",
    "balanced_accuracy": "Balanced ACC",
}


class Experiment(models.Model):
    experiment_id = models.CharField(
        primary_key=True,
        null=False,
        max_length=100
    )

    experiment_name = models.CharField(max_length=200, null=True)

    description = models.CharField(max_length=10000, null=True)

    problem_type = models.CharField(max_length=100, null=True)

    target = models.ForeignKey(
        'FileMetadata',
        on_delete=models.CASCADE,
        null=True,
    )

    features = models.CharField(max_length=1000, null=True)

    split_ratio = models.CharField(max_length=1000, null=True)

    score = models.CharField(max_length=200, null=True)

    algorithm_presets = models.CharField(max_length=200, null=True)

    create_datetime = models.DateTimeField(auto_now_add=True, null=True)

    is_delete = models.BooleanField(null=True)

    task = models.ForeignKey(
        'Task',
        on_delete=models.CASCADE,
        null=True,
    )

    file = models.ForeignKey(
        'File',
        on_delete=models.CASCADE,
        null=True,
    )

    best_model_id = models.CharField(max_length=200, null=True)

    @staticmethod
    def get_experiments():
        return Experiment.objects.filter(is_delete__exact=False).order_by('-create_datetime')

    @staticmethod
    def get_experiment_api(experiment_id):
        try:
            experiment_obj = Experiment.objects.get(pk=experiment_id)
        except Experiment.DoesNotExist:
            return {'code': 404, 'description': 'Experiment id does not exist'}

        # Convert features id into features name
        features_id = [int(x) for x in experiment_obj.features.split(",")]
        features_name = FileMetadata.get_column_name_list(experiment_obj.file.file_id, features_id)

        experiment_dict = {
            "experiment_id": experiment_obj.experiment_id,
            "experiment_name": experiment_obj.experiment_name,

            "file_id": experiment_obj.file.file_id,

            "target": experiment_obj.target.column_name,
            "features": features_name,

            "problem_type": experiment_obj.problem_type,
            "score": experiment_obj.score,
            "split_ratio": 7 if experiment_obj.split_ratio is None else int(experiment_obj.split_ratio),
        }
        return {'code': 200, 'description': 'Success', 'result': experiment_dict}

    @staticmethod
    def get_experiment_ui(pk):
        experiment_obj = Experiment.objects.get(pk=pk)
        experiment_obj.score = SCORE.get(experiment_obj.score, experiment_obj.score)
        return experiment_obj


class File(models.Model):
    file_id = models.CharField(
        primary_key=True,
        null=False,
        max_length=100
    )
    file_name = models.CharField(max_length=200, null=True)

    file_path = models.CharField(max_length=200, null=True)

    is_delete = models.BooleanField(null=True)
    
    is_external = models.BooleanField(null=True, default=False)

    create_datetime = models.DateTimeField(auto_now_add=True, null=True)

    task = models.ForeignKey(
        'Task',
        on_delete=models.CASCADE,
        null=True,
    )

    file_eda = models.ForeignKey(
        'FileEda',
        on_delete=models.CASCADE,
        null=True,
    )

    @staticmethod
    def get_files():
        return File.objects.filter(is_delete__exact=False).order_by('-create_datetime')

    @staticmethod
    def get_success_files():
        return File.objects.filter(is_delete__exact=False).filter(task__status__exact=2).order_by('-create_datetime')


class FileEda(models.Model):
    file_eda_id = models.AutoField(
        primary_key=True
    )

    task = models.ForeignKey(
        'Task',
        on_delete=models.CASCADE,
        null=True,
    )


class FileMetadata(models.Model):
    file_metadata_id = models.AutoField(
        primary_key=True
    )

    column_name = models.CharField(max_length=200, null=True)

    data_type = models.CharField(max_length=200, null=True)

    is_category = models.BooleanField(null=True)

    describe = models.JSONField(null=True)

    file = models.ForeignKey(
        'File',
        on_delete=models.CASCADE,
        null=True,
    )

    @staticmethod
    def get_column_name_list(file_id, column_id_list):
        file_meta = FileMetadata.objects.filter(file_id__exact=file_id).filter(pk__in=column_id_list)

        cols = [f.column_name for f in file_meta]
        return cols


class Model(models.Model):
    model_id = models.CharField(
        primary_key=True,
        null=False,
        max_length=100
    )

    model_type = models.CharField(max_length=300, null=True)

    model_name = models.CharField(max_length=300, null=True)

    score_val = models.FloatField(null=True)

    score_test = models.FloatField(null=True)

    fit_time = models.FloatField(null=True)

    predict_time = models.FloatField(null=True)

    feature_importance = models.JSONField(null=True)

    stack_level = models.FloatField(null=True)

    experiment = models.ForeignKey(
        'Experiment',
        on_delete=models.CASCADE,
        null=True,
    )

    predict_label_encode = models.JSONField(null=True)

    @staticmethod
    def create_model_id(experiment_id, model_name):
        return "{}_{}".format(experiment_id, model_name)


class Predict(models.Model):
    predict_id = models.CharField(
        primary_key=True,
        null=False,
        max_length=300
    )

    predict_name = models.CharField(max_length=200, null=True)

    description = models.CharField(max_length=10000, null=True)

    create_datetime = models.DateTimeField(auto_now_add=True, null=True)

    is_delete = models.BooleanField(null=True)

    file = models.ForeignKey(
        'File',
        on_delete=models.CASCADE,
        null=True,
    )

    experiment_id = models.CharField(max_length=200, null=True)

    model = models.ForeignKey(
        'Model',
        on_delete=models.CASCADE,
        null=True,
    )

    task = models.ForeignKey(
        'Task',
        on_delete=models.CASCADE,
        null=True,
    )

    @staticmethod
    def get_predict(experiment_id):
        return Predict.objects.filter(experiment_id=experiment_id).filter(is_delete=False)

    @staticmethod
    def get_predict_api(predict_id):
        try:
            predict_obj = Predict.objects.get(pk=predict_id)
        except Predict.DoesNotExist:
            return {'code': 404, 'description': 'Predict id does not exist'}

        experiment_obj = predict_obj.model.experiment
        predict_dict = {
            "model_name": predict_obj.model.model_name,
            "file_id": predict_obj.file.file_id,
            "experiment": {
                "experiment_id": experiment_obj.experiment_id,
                "problem_type": experiment_obj.problem_type
            }
        }
        return {'code': 200, 'description': 'Success', 'result': predict_dict}


class Evaluation(models.Model):
    evaluation_id = models.CharField(
        primary_key=True,
        null=False,
        max_length=100
    )

    evaluation_name = models.CharField(
        null=True,
        max_length=100
    )

    file = models.ForeignKey(
        'File',
        on_delete=models.CASCADE,
        null=True,
    )

    model = models.ForeignKey(
        'Model',
        on_delete=models.CASCADE,
        null=True,
    )

    task = models.ForeignKey(
        'Task',
        on_delete=models.CASCADE,
        null=True,
    )

    create_datetime = models.DateTimeField(auto_now_add=True, null=True)

    is_delete = models.BooleanField(null=True)

    scores = models.JSONField(null=True)

    confusion = models.JSONField(null=True)

    @staticmethod
    def get_evaluation_api(evaluation_id):

        try:
            evaluation_obj = Evaluation.objects.get(pk=evaluation_id)
        except Evaluation.DoesNotExist:
            return {'code': 404, 'description': 'Evaluation id does not exist'}

        evaluation_info = {
            "experiment_id": evaluation_obj.model.experiment.experiment_id,
            "model_name": evaluation_obj.model.model_name,
            "file_id": evaluation_obj.file.file_id if evaluation_obj.file is not None else None
        }

        return {'code': 200, 'description': 'Success', 'result': evaluation_info}

    @staticmethod
    def get_default_evaluation(model_id):
        evaluation_obj = Evaluation.objects.filter(model_id__exact=model_id).filter(file=None)

        if len(evaluation_obj) >= 1:
            evaluation_obj = evaluation_obj[0]
        else:
            evaluation_obj = None

        return evaluation_obj


class EvaluationPredictActual(models.Model):
    evaluation_predict_actual_id = models.AutoField(
        primary_key=True
    )

    evaluation = models.ForeignKey(
        'Evaluation',
        on_delete=models.CASCADE,
        null=True,
    )

    predict_vs_actual = models.JSONField(null=True)


class EvaluationClass(models.Model):
    evaluation_class_id = models.CharField(
        primary_key=True,
        null=False,
        max_length=100
    )

    evaluation_class_name = models.CharField(max_length=200, null=True)

    evaluation = models.ForeignKey(
        'Evaluation',
        on_delete=models.CASCADE,
        null=True,
    )


class EvaluationClassRocLift(models.Model):
    evaluation_class_roc_lift_id = models.AutoField(
        primary_key=True
    )

    scores = models.JSONField(null=True)

    evaluation_class = models.ForeignKey(
        'EvaluationClass',
        on_delete=models.CASCADE,
        null=True,
    )


class EvaluationClassDistribution(models.Model):
    evaluation_class_distribution_id = models.AutoField(
        primary_key=True
    )

    evaluation_class = models.ForeignKey(
        'EvaluationClass',
        on_delete=models.CASCADE,
        null=True,
    )

    predict_distribution = models.JSONField(null=True)


class EvaluationSubPopulation(models.Model):
    sub_population_id = models.CharField(
        primary_key=True,
        null=False,
        max_length=100
    )

    file_metadata = models.ForeignKey(
        'FileMetadata',
        on_delete=models.CASCADE,
        null=True,
    )

    evaluation = models.ForeignKey(
        'Evaluation',
        on_delete=models.CASCADE,
        null=True,
    )

    task = models.ForeignKey(
        'Task',
        on_delete=models.CASCADE,
        null=True,
    )

    sub_population = models.JSONField(null=True)


class Task(models.Model):
    task_id = models.CharField(
        primary_key=True,
        null=False,
        max_length=100
    )

    status = models.IntegerField(null=True)

    create_datetime = models.DateTimeField(auto_now_add=True, null=True)

    start_datetime = models.DateTimeField(null=True)

    finish_datetime = models.DateTimeField(null=True)

    description = models.CharField(max_length=10000, null=True)

    def as_json(self):
        return dict(
            task_id=self.task_id, status=self.status
        )


class BulkCreateManager(object):
    """
    This helper class keeps track of ORM objects to be created for multiple
    model classes, and automatically creates those objects with `bulk_create`
    when the number of objects accumulated for a given model class exceeds
    `chunk_size`.
    Upon completion of the loop that's `add()`ing objects, the developer must
    call `done()` to ensure the final set of objects is created for all models.
    """

    def __init__(self, chunk_size=100):
        self._create_queues = defaultdict(list)
        self.chunk_size = chunk_size

    def _commit(self, model_class):
        model_key = model_class._meta.label
        model_class.objects.bulk_create(self._create_queues[model_key])
        self._create_queues[model_key] = []

    def add(self, obj):
        """
        Add an object to the queue to be created, and call bulk_create if we
        have enough objs.
        """
        model_class: object = type(obj)
        model_key = model_class._meta.label
        self._create_queues[model_key].append(obj)
        if len(self._create_queues[model_key]) >= self.chunk_size:
            self._commit(model_class)

    def done(self):
        """
        Always call this upon completion to make sure the final partial chunk
        is saved.
        """
        for model_name, objs in self._create_queues.items():
            if len(objs) > 0:
                self._commit(apps.get_model(model_name))


class Explain(models.Model):
    explain_id = models.CharField(
        primary_key=True,
        null=False,
        max_length=100
    )

    task = models.ForeignKey(
        'Task',
        on_delete=models.CASCADE,
        null=True,
    )

    evaluation = models.ForeignKey(
        'Evaluation',
        on_delete=models.CASCADE,
        null=True,
    )

    @staticmethod
    def get_explain_api(explain_id):
        try:
            explain_obj = Explain.objects.get(pk=explain_id)
        except Explain.DoesNotExist:
            return {'code': 404, 'description': 'Experiment id does not exist'}

        file_id = None if explain_obj.evaluation.file is None else explain_obj.evaluation.file.file_id
        explain_info = {
            "experiment_id": explain_obj.evaluation.model.experiment.experiment_id,
            "file_id": file_id,
            "model_name": explain_obj.evaluation.model.model_name,
        }

        return {'code': 200, 'description': 'Success', 'result': explain_info}


class ExplainPdp(models.Model):
    explain_pdp_id = models.CharField(
        primary_key=True,
        null=False,
        max_length=100
    )

    explain = models.ForeignKey(
        'Explain',
        on_delete=models.CASCADE,
        null=True,
    )

    feature = models.CharField(max_length=100, null=True)


class ExplainPdpRegress(models.Model):
    explain_pdp_regress_id = models.AutoField(
        primary_key=True
    )

    explain_pdp = models.ForeignKey(
        'ExplainPdp',
        on_delete=models.CASCADE,
        null=True,
    )

    pdp_values = models.JSONField(null=True)


class ExplainPdpClass(models.Model):
    explain_pdp_class_id = models.CharField(
        primary_key=True,
        null=False,
        max_length=100
    )

    explain_pdp = models.ForeignKey(
        'ExplainPdp',
        on_delete=models.CASCADE,
        null=True,
    )

    class_name = models.CharField(max_length=100, null=True)


class ExplainPdpClassValues(models.Model):
    pdp_id = models.AutoField(
        primary_key=True
    )

    explain_pdp_class = models.ForeignKey(
        'ExplainPdpClass',
        on_delete=models.CASCADE,
        null=True,
    )

    pdp_values = models.JSONField(null=True)


class Connection(models.Model):
    connection_id = models.AutoField(
        primary_key=True
    )

    connection_name = models.CharField(
        max_length=100
    )

    database = models.CharField(
        max_length=100
    )

    host = models.CharField(
        max_length=500
    )

    user_name = models.CharField(
        max_length=200
    )

    password = models.BinaryField()

    is_delete = models.BooleanField(null=True)

    create_datetime = models.DateTimeField(auto_now_add=True, null=True)

    @staticmethod
    def encrypt_password(password):
        secret_key = settings.SECRET_KEY_FOR_ENCRYPTION
        cipher_suite = Fernet(secret_key)
        encrypted_password = cipher_suite.encrypt(str(password).encode("utf-8"))

        return encrypted_password

    def decrypt_password(self):
        secret_key = settings.SECRET_KEY_FOR_ENCRYPTION
        cipher_suite = Fernet(secret_key)
        decrypted_password = cipher_suite.decrypt(self.password.tobytes())
        return decrypted_password.decode()

    @staticmethod
    def get_connections():
        connection_objs = Connection.objects.filter(is_delete__exact=False)

        _connection_objs = []
        for connection_obj in connection_objs:
            connection_obj.password = connection_obj.decrypt_password()

            _connection_objs.append(connection_obj)

        return _connection_objs

    @staticmethod
    def get_connection(connection_id, decrypt_password=True):
        try:
            connection_obj = Connection.objects.get(pk=connection_id)
            if decrypt_password:
                connection_obj.password = connection_obj.decrypt_password()
            else:
                # Convert password into string
                connection_obj.password = base64.b64encode(connection_obj.password).decode('utf-8')
        except Experiment.DoesNotExist:
            return {'code': 404, 'description': 'Experiment id does not exist'}

        return connection_obj

    def get_tables_from_db(self):

        try:
            connection = teradatasql.connect(
                host=self.host,
                user=self.user_name,
                password=self.password
            )

            cursor = connection.cursor()

            # Execute a SQL query to retrieve table names
            sql_query = "SELECT TableName FROM DBC.TablesV WHERE DatabaseName = '{}';".format(self.database)
            cursor.execute(sql_query)

            # Fetch and print the table names
            table_names = [row[0] for row in cursor.fetchall()]

            cursor.close()
            connection.close()

            return {
                'code': 200, 'result': {
                    "table_names": table_names
                }

            }
        except Exception as e:
            return {'code': 405, 'description': "Can't connect to DB, {}".format(e)}

    def as_json(self):
        return dict(
            host_name=self.host, username=self.user_name,
            password=self.decrypt_password(),  # TODO: Not decrypt at here
            database=self.database
        )


class Deployment(models.Model):
    deploy_id = models.CharField(
        primary_key=True,
        null=False,
        max_length=300
    )

    deploy_name = models.CharField(
        max_length=100
    )

    model = models.ForeignKey(
        'Model',
        on_delete=models.CASCADE,
        null=True,
    )

    teradata_model_id = models.CharField(
        max_length=100,
        null=True,
    )

    task = models.ForeignKey(
        'Task',
        on_delete=models.CASCADE,
        null=True,
    )

    experiment_id = models.CharField(max_length=200, null=True)

    connection = models.ForeignKey(
        'Connection',
        on_delete=models.CASCADE,
        null=True,
    )

    create_datetime = models.DateTimeField(auto_now_add=True, null=True)

    is_delete = models.BooleanField(null=True, default=False)

    output_table = models.CharField(
        max_length=100
    )

    sql_preprocess = models.CharField(
        max_length=10000
    )

    sql_prediction = models.CharField(
        max_length=10000
    )

    @staticmethod
    def get_deployments(experiment_id):
        deployment_objs = Deployment.objects.filter(experiment_id=experiment_id).filter(is_delete__exact=False).\
            order_by('-create_datetime')
        return deployment_objs

    def as_json(self):
        return dict(
            deploy_id=self.deploy_id, deploy_name=self.deploy_name, output_table=self.output_table,
            experiment_id=self.model.experiment.experiment_id,
            model_name=self.model.model_name,
            teradata_model_id=self.teradata_model_id
        )

    @staticmethod
    def get_deploy_api(deploy_id):
        try:
            deploy_obj = Deployment.objects.get(pk=deploy_id)
        except Explain.DoesNotExist:
            return {'code': 404, 'description': 'Experiment id does not exist'}

        deploy_info = {
            "deploy_info": deploy_obj.as_json(),
            "connection_info": deploy_obj.connection.as_json(),
        }

        return {'code': 200, 'description': 'Success', 'result': deploy_info}
