from celery.decorators import task
from utils.transform import Input, Output, get_file_eda_url, get_file_url
from utils import config, services
from jobs.automl import Train, Predict, Evaluation, Explain
from pandas_profiling import ProfileReport


@task(name="upload_file", bind=True)
def upload_file_task(self, file_id):
    task_id = self.request.id

    _data = {
        "task_id": task_id,
        "status": config.TASK_STATUS.get('STARTED'),
    }

    services.post(data=_data, target_path=config.TARGET_PATH.get("task"))

    # TODO :
    # 1. Check column name, remove special character
    # 2. Add data type, receive from  end-user
    # 3. Save file as pickle or parquet file
    try:
        # Load file
        file_url = get_file_url(file_id)

        df_file = Input(file_url).from_csv()

        # Create metadata
        file_metadata_dict = df_file.dtypes.apply(lambda x: str(x)).to_dict()

        # Post metadata into server
        _data = {
            "file_id": file_id,
            "file_metadata_dict": file_metadata_dict
        }
        services.post(data=_data, target_path=config.TARGET_PATH.get("file_metadata"))

        # # Post data into server
        _data = {
            "task_id": task_id,
            "status": config.TASK_STATUS.get('SUCCESS'),
        }

        services.post(data=_data, target_path=config.TARGET_PATH.get("task"))

    except Exception as e:
        mes = "Upload file failure. Error {}".format(e)
        _data = {
            "task_id": task_id,
            "status": config.TASK_STATUS.get('FAILURE'),
            "description": mes[:300]
        }

        services.post(data=_data, target_path=config.TARGET_PATH.get("task"))
        raise Exception(mes[:300])


@task(name="create_file_eda_task", bind=True)
def create_file_eda_task(self, file_id):
    task_id = self.request.id

    _data = {
        "task_id": task_id,
        "status": config.TASK_STATUS.get('STARTED'),
    }
    services.post(data=_data, target_path=config.TARGET_PATH.get("task"))

    # Step 1 : Load file
    try:
        file_url = get_file_url(file_id)

        df_file = Input(file_url).from_csv()

        profile = ProfileReport(df_file, title="", explorative=False, minimal=False)

        url = get_file_eda_url(file_id)
        profile.to_file(url)

        _data = {
            "task_id": task_id,
            "status": config.TASK_STATUS.get('SUCCESS'),
        }

        services.post(data=_data, target_path=config.TARGET_PATH.get("task"))
    except Exception as e:
        mes = "Create experiment failure. Error {}".format(e)
        _data = {
            "task_id": task_id,
            "status": config.TASK_STATUS.get('FAILURE'),
            "description": mes[:300]
        }

        services.post(data=_data, target_path=config.TARGET_PATH.get("task"))

        raise Exception(mes[:300])


@task(name="create_experiment_task", bind=True)
def create_experiment_task(self, experiment_id):
    task_id = self.request.id

    _data = {
        "task_id": task_id,
        "status": config.TASK_STATUS.get('STARTED'),
    }
    services.post(data=_data, target_path=config.TARGET_PATH.get("task"))

    try:
        _models = Train(experiment_id)
        _models.train()

        _data = {
            "task_id": task_id,
            "status": config.TASK_STATUS.get('SUCCESS'),
        }

        services.post(data=_data, target_path=config.TARGET_PATH.get("task"))

    except Exception as e:
        mes = "Create experiment failure. Error {}".format(e)
        _data = {
            "task_id": task_id,
            "status": config.TASK_STATUS.get('FAILURE'),
            "description": mes[:300]
        }

        services.post(data=_data, target_path=config.TARGET_PATH.get("task"))

        raise Exception(mes[:300])


@task(name="predict_task", bind=True)
def predict_task(self, predict_id):
    task_id = self.request.id

    _data = {
        "task_id": task_id,
        "status": config.TASK_STATUS.get('STARTED'),
    }
    services.post(data=_data, target_path=config.TARGET_PATH.get("task"))

    try:
        _predict = Predict(predict_id)
        _predict.predict()

        _data = {
            "task_id": task_id,
            "status": config.TASK_STATUS.get('SUCCESS'),
        }

        services.post(data=_data, target_path=config.TARGET_PATH.get("task"))

    except Exception as e:
        mes = "Create experiment failure. Error {}".format(e)
        _data = {
            "task_id": task_id,
            "status": config.TASK_STATUS.get('FAILURE'),
            "description": mes[:300]
        }

        services.post(data=_data, target_path=config.TARGET_PATH.get("task"))

        raise Exception(mes[:300])


@task(name="evaluation_task", bind=True)
def evaluation_task(self, evaluation_id):
    task_id = self.request.id

    _data = {
        "task_id": task_id,
        "status": config.TASK_STATUS.get('STARTED'),
    }
    services.post(data=_data, target_path=config.TARGET_PATH.get("task"))

    try:
        _evaluation = Evaluation(evaluation_id)
        _evaluation.evaluate()

        _data = {
            "task_id": task_id,
            "status": config.TASK_STATUS.get('SUCCESS'),
        }

        services.post(data=_data, target_path=config.TARGET_PATH.get("task"))

    except Exception as e:
        mes = "Create experiment failure. Error {}".format(e)
        _data = {
            "task_id": task_id,
            "status": config.TASK_STATUS.get('FAILURE'),
            "description": mes[:300]
        }

        services.post(data=_data, target_path=config.TARGET_PATH.get("task"))

        raise Exception(mes[:300])


@task(name="evaluation_sub_population_task", bind=True)
def evaluation_sub_population_task(self, evaluation_id, sub_population_id, column_name):
    task_id = self.request.id

    _data = {
        "task_id": task_id,
        "status": config.TASK_STATUS.get('STARTED'),
    }
    services.post(data=_data, target_path=config.TARGET_PATH.get("task"))

    try:
        _evaluation = Evaluation(evaluation_id)
        _evaluation.sub_population(sub_population_id, column_name)

        _data = {
            "task_id": task_id,
            "status": config.TASK_STATUS.get('SUCCESS'),
        }

        services.post(data=_data, target_path=config.TARGET_PATH.get("task"))

    except Exception as e:
        mes = "Create experiment failure. Error {}".format(e)
        _data = {
            "task_id": task_id,
            "status": config.TASK_STATUS.get('FAILURE'),
            "description": mes[:300]
        }

        services.post(data=_data, target_path=config.TARGET_PATH.get("task"))

        raise Exception(mes[:300])


@task(name="explain_task", bind=True)
def explain_task(self, explain_id):
    task_id = self.request.id

    _data = {
        "task_id": task_id,
        "status": config.TASK_STATUS.get('STARTED'),
    }
    services.post(data=_data, target_path=config.TARGET_PATH.get("task"))

    try:
        explain = Explain(explain_id)
        explain.explain()

        _data = {
            "task_id": task_id,
            "status": config.TASK_STATUS.get('SUCCESS'),
        }

        services.post(data=_data, target_path=config.TARGET_PATH.get("task"))

    except Exception as e:
        mes = "Create explain failure. Error {}".format(e)
        _data = {
            "task_id": task_id,
            "status": config.TASK_STATUS.get('FAILURE'),
            "description": mes[:300]
        }

        services.post(data=_data, target_path=config.TARGET_PATH.get("task"))

        raise Exception(mes[:300])