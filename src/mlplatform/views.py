from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Experiment, Predict, Model, Task, File, BulkCreateManager, FileMetadata, FileEda, Evaluation, SCORE
from django.db.models import Max
from django.core.files.storage import default_storage
from django.urls import reverse
from django.http import HttpResponseRedirect
import uuid
from django.views.decorators.csrf import csrf_exempt
from jobs.tasks import upload_file_task, create_file_eda_task, create_experiment_task, predict_task, evaluation_task
from utils.transform import get_file_url, get_predict_url, Input, get_file_eda_url
from utils import config
import json
from celery.task.control import revoke
from django.http.response import JsonResponse
from datetime import datetime
from django.http import HttpResponse
import numpy as np


def index(request):

    if request.POST.get("delete"):
        delete_experiment_id = request.POST.get("delete_experiment_id")

        experiment_obj = Experiment.objects.get(pk=delete_experiment_id)
        experiment_obj.is_delete = True
        experiment_obj.save()

        return redirect("index")

    experiment_objs = Experiment.get_experiments()
    context = {
        "experiment_objs": experiment_objs
    }
    return render(request, 'index.html', context=context)


@csrf_exempt
def experiment(request):
    """
    API used to call from worker
    :param request:
    :return:
    """
    if request.method == "GET":
        experiment_id = request.GET.get('experiment_id', None)
        return JsonResponse(Experiment.get_experiment_api(experiment_id))

    if request.method == "POST":
        body_unicode = request.body.decode('utf-8')
        agrs = json.loads(body_unicode)

        experiment_id = agrs.get("experiment_id", None)

        if experiment_id is not None:
            try:
                experiment_obj = Experiment.objects.get(pk=experiment_id)

                experiment_obj.best_model_id = agrs.get("best_model_id", None)
                experiment_obj.save()
            except Exception as e:
                return JsonResponse({'code': 400, 'description': "Update experiment fail. Error : {}".format(e[:300])})

        return JsonResponse({'code': 200, 'description': "success"})


@csrf_exempt
def model(request):
    """
    API request from worker.
    :param request:
    :param pk: model_id
    :return:
    """
    if request.method == "POST":
        body_unicode = request.body.decode('utf-8')
        agrs = json.loads(body_unicode)

        experiment_id = agrs.get("experiment_id", None)

        if experiment_id is not None:
            experiment_obj = Experiment.objects.get(pk=experiment_id)

        models_list = agrs.get("models_list", None)

        bulk_mgr = BulkCreateManager(chunk_size=20)
        for _model in models_list:
            model_obj = Model(
                model_id=Model.create_model_id(experiment_id, _model.get("model_name")),
                model_name=_model.get("model_name"),
                score_val=_model.get("score_val"),
                score_test=_model.get("score_test"),
                fit_time=_model.get("fit_time"),
                predict_time=_model.get("predict_time"),
                feature_importance=_model.get("features_importance"),
                experiment=experiment_obj
            )
            bulk_mgr.add(model_obj)
        try:
            bulk_mgr.done()
        except Exception as e:
            mes = 'Can not insert data into db: {}, ERROR : {}'.format(experiment_id, e)
            return JsonResponse({'code': 405, 'description': mes})

        return JsonResponse({'code': 200, 'description': "success"})


@csrf_exempt
def model_detail(request, pk=None):
    """
    Used for frontend
    :param request:
    :param pk: model_id
    :return:
    """
    if request.method == "GET":
        model_obj = Model.objects.get(pk=pk)

        evaluation_obj = Evaluation.get_default_evaluation(pk)
        feature_importance, confusion_matrix, scores = None, None, None

        if evaluation_obj is not None:
            # Adjust score
            if evaluation_obj.task.status == 2:
                scores = dict((SCORE.get(key, key), np.abs(val)) for key, val in evaluation_obj.scores.items())
                scores.pop('median_absolute_error', None)

                # Generate feature importance data
                _f_importance = {k: v for k, v in
                                 sorted(model_obj.feature_importance.items(), key=lambda item: item[1], reverse=True)}

                feature_importance = {
                    "label": (list(_f_importance.keys())),
                    "data": list(_f_importance.values())
                }

                # Generate Confusion matrix data
                if evaluation_obj.confusion is not None:
                    i, j = 0, 0
                    values = []

                    for key, val in evaluation_obj.confusion.items():
                        for sub_key, sub_val in val.items():
                            values.append([i, j, sub_val])
                            j += 1
                        i += 1
                        j = 0

                    confusion_matrix = {
                        "category": list(evaluation_obj.confusion.keys()),
                        "data": values
                    }

        context = {
            "model_obj": model_obj,
            "evaluation_obj": evaluation_obj,
            "scores": scores,
            "feature_importance": feature_importance,
            "confusion_matrix": confusion_matrix,
        }
        return render(request, 'model/detail.html', context=context)

    if request.method == "POST":
        # Run evaluation
        model_id = request.POST.get("submit_evaluation", None)
        file_id = request.POST.get("submit_file_id", None)

        if model_id is not None:
            model_obj = Model.objects.get(pk=model_id)

            file_obj = None if file_id is None else File.objects.get(pk=file_id)

            evaluation_id = str(uuid.uuid1())

            task_id = evaluation_task.delay(evaluation_id)

            task_obj = Task(
                task_id=task_id,
                status=config.TASK_STATUS.get('PENDING')
            )
            task_obj.save()

            evaluation_obj = Evaluation(
                evaluation_id=evaluation_id,
                file=file_obj,
                task=task_obj,
                model=model_obj
            )
            evaluation_obj.save()

            return redirect("model_detail", pk=model_id)


@csrf_exempt
def predict(request):
    """
    API used to call from worker
    :param request:
    :return:
    """
    if request.method == "GET":
        experiment_id = request.GET.get('predict_id', None)
        return JsonResponse(Predict.get_predict_api(experiment_id))


@csrf_exempt
def evaluation(request):
    """
    API used to call from worker
    :param request:
    :return:
    """

    if request.method == "GET":
        evaluation_id = request.GET.get("evaluation_id", None)
        return JsonResponse(Evaluation.get_evaluation_api(evaluation_id))

    if request.method == "POST":
        body_unicode = request.body.decode('utf-8')
        agrs = json.loads(body_unicode)

        evaluation_id = agrs.get("evaluation_id", None)
        scores = agrs.get("scores", None)
        confusion_matrix = agrs.get("confusion_matrix", None)
        predict_vs_actual = agrs.get("predict_vs_actual", None)

        if evaluation_id is not None:

            evaluation_obj = Evaluation.objects.get(pk=evaluation_id)
            evaluation_obj.scores = scores
            evaluation_obj.confusion = confusion_matrix
            evaluation_obj.predict_vs_actual = predict_vs_actual
            evaluation_obj.save()

        return JsonResponse({'code': 200, 'description': "success"})


@csrf_exempt
def experiment_detail(request, pk):

    # Processing create predict
    if request.POST.get("create_predict"):
        file_id = request.POST.get('select_file')
        model_id = request.POST.get('select_model')
        predict_name = request.POST.get('predict_name')

        predict_id = str(uuid.uuid1())
        # Task
        task_id = predict_task.delay(predict_id)

        task_obj = Task(
                task_id=task_id,
                status=config.TASK_STATUS.get('PENDING')
            )
        task_obj.save()

        file_obj = File.objects.get(pk=file_id)
        model_obj = Model.objects.get(pk=model_id)

        predict_obj = Predict(
            predict_id=predict_id,
            predict_name=predict_name,
            file=file_obj,
            model=model_obj,
            is_delete=False,
            experiment_id=pk,
            task=task_obj,
        )
        predict_obj.save()

        return redirect("experiment_detail", pk=pk)

    # Processing delete predict
    if request.POST.get("delete_predict"):
        predict_id = request.POST.get("delete_predict_id")

        predict_obj = Predict.objects.get(pk=predict_id)
        predict_obj.is_delete = True
        predict_obj.save()

        return redirect("experiment_detail", pk=pk)

    # Processing download predict
    if request.POST.get("download_predict"):
        predict_id = request.POST.get("download_predict")

        try:
            predict_obj = Predict.objects.get(pk=predict_id)
        except Exception as e:
            mes = '<h1>Not found predict id.  </h1>. ERROR : {}'.format(e)
            return HttpResponse(mes)

        # Load predict file
        file_url = get_predict_url(predict_id)

        try:
            with open(file_url, 'rb') as fh:
                response = HttpResponse(fh.read(),
                                        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                response['Content-Disposition'] = 'attachment; filename={}.xlsx'.format(predict_obj.predict_name)
                return response
        except Exception as e:
            mes = '<h1>Can not open excel file at local.  </h1>. ERROR : {}'.format(e)
            return HttpResponse(mes)

    # MAIN.
    if request.method == "GET":
        experiment_obj = Experiment.get_experiment_ui(pk)

        context = {
            "experiment_obj": experiment_obj
        }

        if experiment_obj.task.status == 2:

            model_objs = Model.objects.filter(experiment_id__exact=pk)

            predict_objs = Predict.get_predict(pk)

            file_objs = File.get_success_files()

            best_model_obj = Model.objects.get(pk=experiment_obj.best_model_id)

            _f_importance = {k: v for k, v in sorted(best_model_obj.feature_importance.items(), key=lambda item: item[1], reverse=True)}

            feature_importance = {
                "label": (list(_f_importance.keys())),
                "data": list(_f_importance.values())
            }

            models_performance = []
            score_max = model_objs.aggregate(Max("score_val")).get('score_val__max')

            # Generate data to visualize
            for model_obj in model_objs:
                if model_obj.experiment.problem_type == "regression":
                    r = (score_max - model_obj.score_val*0.9)/2
                else:
                    r = model_obj.score_val * 10

                _data = [{"x": model_obj.score_val, "y": model_obj.score_test, "r": r}]

                models_performance.append({"label": [model_obj.model_name], "data": _data,
                                           "backgroundColor": "rgba(60,186,159,0.2)",
                                           "borderColor": "rgba(60,186,159,1)",
                                           })

            context = {
                "experiment_obj": experiment_obj,
                "model_objs": model_objs,
                "predict_objs": predict_objs,
                "file_objs": file_objs,
                "best_model_obj": best_model_obj,
                "feature_importance": feature_importance,
                "models_performance": models_performance
            }

    return render(request, 'experiment/detail.html', context=context)


def experiment_create(request):
    form = {
        "view_id": int(request.POST.get('view_id', 1)),
        "file_id": request.POST.get('file_id', None),
        "target_id": int(request.POST.get('target_id', 0)),
        "features_id": request.POST.get('features_id', None),
        "problem_type": request.POST.get('problem_type', None),
        "score": request.POST.get('score', None),
        "split_ratio": request.POST.get('split_ratio', None),
        "experiment_name": request.POST.get('experiment_name', None),
}

    if request.POST.get('complete'):
        # Run worker task
        experiment_id = str(uuid.uuid1())
        task_id = create_experiment_task.delay(experiment_id)

        task_obj = Task(
            task_id=task_id,
            status=config.TASK_STATUS.get('PENDING')
        )
        task_obj.save()

        file_obj = File.objects.get(pk=form.get("file_id"))

        target_obj = FileMetadata.objects.get(pk=form.get("target_id"))

        # Redirect into detail exp page
        experiment_obj = Experiment(
            experiment_id=experiment_id,
            experiment_name=form.get("experiment_name"),
            problem_type=form.get("problem_type"),
            target=target_obj,
            features=form.get("features_id"),
            file=file_obj,
            score=form.get("score"),
            split_ratio=form.get("split_ratio"),
            is_delete=False,
            task=task_obj
        )
        experiment_obj.save()

        return redirect("experiment_detail", pk=experiment_id)

    file_objs = File.get_files()

    f_metadata_objs = None
    if form.get("file_id") is not None:
        f_metadata_objs = FileMetadata.objects.filter(file_id__exact=form.get("file_id"))

    context = {
        'form': form,
        'file_objs': file_objs,
        'f_metadata_objs': f_metadata_objs
    }

    return render(request, 'experiment/create.html', context=context)


@csrf_exempt
def file(request):

    if request.method == 'GET':
        file_objs = File.get_files()

        context = {
            'file_objs': file_objs
        }
        return render(request, 'file/index.html', context=context)

    if request.method == 'POST':
        # 1. Upload file
        if request.FILES.get('upload_file', None) is not None:
            uploaded_file = request.FILES['upload_file']

            if str(uploaded_file).split(".")[-1] != "csv":
                messages.add_message(request, messages.ERROR, "Only accept csv file")
                return HttpResponseRedirect(reverse("file"))

            file_id = uuid.uuid1()
            file_path = get_file_url(file_id)

            # Save file into local
            default_storage.save(file_path, uploaded_file)

            # TODO : Return file metadata
            df_file = Input(file_path).from_csv()
            print(df_file.columns, df_file.dtypes)

            # Cal worker run task
            task_id = upload_file_task.delay(file_id)
            task_id = str(task_id)

            task_obj = Task(
                task_id=task_id,
                status=config.TASK_STATUS.get('PENDING')
            )
            task_obj.save()

            # Save file info
            file_obj = File(
                file_id=file_id, file_name='{}'.format(uploaded_file),
                file_path=file_path, is_delete=False, task=task_obj
            )
            file_obj.save()

        # 2. Soft Delete file
        _delete_file_id = request.POST.get('delete_file_id', None)
        if _delete_file_id is not None:
            file_obj = File.objects.get(pk=_delete_file_id)

            if file_obj is not None:
                file_obj.is_delete = True
            file_obj.save()

        # 3. crete file metadata
        file_id = request.POST.get('create_file_eda', None)
        if file_id is not None:
            file_obj = File.objects.get(pk=file_id)

            task_id = create_file_eda_task.delay(file_id)

            task_obj = Task(
                task_id=task_id,
                status=config.TASK_STATUS.get('PENDING')
            )
            task_obj.save()

            file_eda_obj = FileEda(
                task=task_obj,
            )
            file_eda_obj.save()

            file_obj.file_eda = file_eda_obj
            file_obj.save()

        return HttpResponseRedirect(reverse("file"))


@csrf_exempt
def file_metadata(request):
    """
    API. Used to update metadata from Worker
    :param request:
    :return:
    """
    if request.method == "POST":
        body_unicode = request.body.decode('utf-8')
        agrs = json.loads(body_unicode)

        file_id, file_metadata_dict = agrs.get("file_id", None), agrs.get("file_metadata_dict", None)

        if (file_id is None) or (file_metadata_dict is None):
            return JsonResponse({'code': 404, 'description': 'File id or Models must be not none'})

        file_obj = File.objects.get(pk=file_id)
        if file_obj is None:
            return JsonResponse({'code': 404, 'description': 'File id were not found'})

        bulk_mgr = BulkCreateManager(chunk_size=20)
        for f_col, f_type in file_metadata_dict.items():
            _is_category = f_type == "object"
            f_metadata_obj = FileMetadata(
                file=file_obj,
                column_name=f_col,
                data_type=f_type,
                is_category=_is_category
            )
            bulk_mgr.add(f_metadata_obj)

        try:
            bulk_mgr.done()
        except Exception as e:
            mes = 'Can not insert data into db: {}, ERROR : {}'.format(file_id, e)
            return JsonResponse({'code': 405, 'description': mes})

        return JsonResponse({'code': 200, 'description': "Success"})

    return JsonResponse({'code': 404, 'description': 'Only support methods POST AND GET'})


@csrf_exempt
def file_eda(request, pk):

    if request.method == 'GET':

        url = "{}{}.html".format("file/eda/", pk)

        return render(request, url)


@csrf_exempt
def task(request, pk=None):
    """
    API. Used to update Task status.
    :param request:
        - task_id : Created when experiment run
        - status id : 'PENDING': 0, 'STARTED': 1, 'SUCCESS': 2, 'FAILURE': 3, 'RETRY': 4, 'REVOKED': 5
        - description
    :return:
    """
    if request.method == "POST":

        body_unicode = request.body.decode('utf-8')
        agrs = json.loads(body_unicode)

        task_id = agrs.get('task_id', None)
        if task_id is None:
            return JsonResponse({'code': 404, 'description': 'Task id must be not none'})

        try:
            task_obj = Task.objects.get(pk=task_id)
        except Task.DoesNotExist:
            return JsonResponse({'code': 405, 'description': 'Task not found'})

        # REVOKED TASK
        if agrs.get('status') == config.TASK_STATUS.get('REVOKED'):
            revoke(agrs.get('task_id'), terminate=True, signal="SIGKILL")

        # Update Task status info
        task_obj.status = agrs.get('status')
        task_obj.description = agrs.get('description', None)

        if agrs.get('status') == config.TASK_STATUS.get('STARTED'):
            task_obj.start_datetime = datetime.now()

        if agrs.get('status') in [config.TASK_STATUS.get('SUCCESS'), config.TASK_STATUS.get('FAILURE'),
                                  config.TASK_STATUS.get('RETRY'), config.TASK_STATUS.get('REVOKED')]:
            task_obj.finish_datetime = datetime.now()
            task_obj.description = agrs.get('description', None)

        try:
            task_obj.save()
        except Exception as e:
            return JsonResponse({'code': 404, 'description': 'Can not update task status. ERROR : {}'.format(agrs, e)})

        return JsonResponse({'code': 200, 'description': 'Success'})

    if request.method == "GET":
        task_obj = Task.objects.get(pk=pk)

        context = {
            'task_obj': task_obj,
        }

        return render(request, 'task/detail.html', context=context)

    return JsonResponse({'code': 404, 'description': 'Method not valid'})
