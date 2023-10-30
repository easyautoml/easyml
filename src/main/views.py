import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Experiment, Predict, Model, Task, File, BulkCreateManager, FileMetadata, FileEda, Evaluation, \
    SCORE, EvaluationClassRocLift, EvaluationClass, EvaluationClassDistribution, EvaluationPredictActual, \
    EvaluationSubPopulation, Explain, ExplainPdp, ExplainPdpRegress, ExplainPdpClass, ExplainPdpClassValues, \
    Connection, Deployment
from django.db.models import Max
from django.core.files.storage import default_storage
from django.urls import reverse
from django.http import HttpResponseRedirect
import uuid
from django.views.decorators.csrf import csrf_exempt
from jobs.tasks import upload_file_task, create_file_eda_task, create_experiment_task, predict_task, evaluation_task, \
    evaluation_sub_population_task, explain_task, ingest_file_from_db_task, deploy_task
from utils.transform import get_file_url, get_predict_url, Input, get_file_eda_url, get_evaluation_url
from utils import config
import json
from celery.task.control import revoke
from django.http.response import JsonResponse
from datetime import datetime
from django.http import HttpResponse
import numpy as np
from django.http import Http404


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

        selected_page = request.GET.get("form_selected_page", 1)

        evaluation_obj = Evaluation.get_default_evaluation(pk)

        file_metadata_objs = FileMetadata.objects.filter(file_id__exact=model_obj.experiment.file.file_id)

        context = {
            "model_obj": model_obj,
            "evaluation_obj": evaluation_obj,
            "selected_page": selected_page,
            "file_metadata_objs": file_metadata_objs,
        }

        # Evaluation still not running
        if evaluation_obj is not None:
            if evaluation_obj.task.status == 2:
                # TODO : MOVE ALL THIS CODE TO MODELS
                # 1. Base score
                scores = dict((SCORE.get(key, key), np.abs(val)) for key, val in evaluation_obj.scores.items())
                scores.pop('median_absolute_error', None)

                # 2. Feature Importance
                _f_importance = {k: v for k, v in
                                 sorted(model_obj.feature_importance.items(), key=lambda item: item[1], reverse=True)}

                feature_importance = {
                    "label": (list(_f_importance.keys())),
                    "data": list(_f_importance.values())
                }

                context.update({
                    "scores": scores,
                    "feature_importance": feature_importance,
                })

                # 3. Lift chart or Residual chart
                if evaluation_obj.model.experiment.problem_type in ["binary", "multiclass"]:

                    # GET ROC, LIFT, and PREDICT PROBABILITY for each class

                    evaluation_class_objs = EvaluationClass.objects.filter(evaluation_id__exact=
                                                                           evaluation_obj.evaluation_id)

                    # Get first class id and class name in first time load the page.
                    evaluation_class_id = request.GET.get("form_selected_class_id",
                                                          evaluation_class_objs[0].evaluation_class_id)
                    selected_class_name = request.GET.get("form_selected_class_name",
                                                          evaluation_class_objs[0].evaluation_class_name)

                    eval_class_roc_lift_obj = EvaluationClassRocLift.objects.filter(
                        evaluation_class_id__exact=evaluation_class_id)[0]

                    eval_class_dist_obj = EvaluationClassDistribution.objects.filter(evaluation_class_id__exact=
                                                                                     evaluation_class_id)[0]

                    df_score = pd.DataFrame(eval_class_roc_lift_obj.scores)[["tpr", "fpr", "overall_population",
                                                                             "target_population"]]
                    # ROC
                    roc = df_score.apply(lambda x: {"x": 1 - x["fpr"], "y": x["tpr"]}, axis=1).to_list()

                    # LIFT
                    lift = df_score.apply(lambda x: {"x": x["overall_population"], "y": x["target_population"]},
                                          axis=1).to_list()

                    # PREDICT PROBABILITY
                    df_predict_dis = pd.DataFrame(eval_class_dist_obj.predict_distribution)
                    _target_class = df_predict_dis.apply(lambda x: {"x": x["predict_prob"],
                                                                    "y": x["target_class_density"]}, axis=1).to_list()
                    _left_class = df_predict_dis.apply(lambda x: {"x": x["predict_prob"],
                                                                  "y": x["left_class_density"]}, axis=1).to_list()

                    predict_distribution = {
                        "target_class": _target_class,
                        "left_class": _left_class
                    }

                    # Score of class
                    class_scores = json.dumps(eval_class_roc_lift_obj.scores)

                    context.update({
                        "evaluation_class_objs": evaluation_class_objs,
                        "selected_class_name": selected_class_name,
                        "roc_chart_data": roc,
                        "lift_chart_data": lift,
                        "predict_distribution": predict_distribution,
                        "class_scores": class_scores,
                    })
                else:
                    eval_predict_actual_obj = EvaluationPredictActual.objects.filter(evaluation_id__exact=
                                                                                     evaluation_obj.evaluation_id)[0]

                    df = pd.DataFrame(eval_predict_actual_obj.predict_vs_actual)

                    df["predict_vs_actual"] = df.apply(lambda x: {"x": x["predict"], "y": x["actual"]}, axis=1)
                    df["residual"] = df.apply(lambda x: {"x": x["predict"], "y": x["predict"] - x["actual"]}, axis=1)

                    context.update({
                        "predict_vs_actual": df.predict_vs_actual.tolist(),
                        "residual": df.residual.tolist(),
                    })

                # 4. Sub Population chart

                column_id = request.GET.get("form_selected_column_id", file_metadata_objs[0].file_metadata_id)
                column_name = request.GET.get("form_selected_column_name", file_metadata_objs[0].column_name)

                is_submit_sub_population = request.GET.get("form_calculate_sub_population", None)

                context.update({
                    "selected_column_id": column_id,
                    "selected_column_name": column_name,
                })

                sub_population_obj = EvaluationSubPopulation.objects.filter(file_metadata_id__exact=int(column_id)).filter(evaluation_id__exact=evaluation_obj.evaluation_id)

                if (not sub_population_obj.exists()) & (is_submit_sub_population is not None):

                    # Call calculation subpopulation chart
                    file_metadata_obj = FileMetadata.objects.get(pk=column_id)

                    sub_population_id = "{}_{}".format(evaluation_obj.evaluation_id, column_id)

                    # evaluation_sub_population_task(self, evaluation_id, sub_population_id, column_name)
                    task_id = evaluation_sub_population_task.delay(evaluation_obj.evaluation_id, sub_population_id,
                                                                   column_name
                                                                   )

                    task_obj = Task(
                        task_id=task_id,
                        status=config.TASK_STATUS.get('PENDING')
                    )
                    task_obj.save()

                    sub_population_obj = EvaluationSubPopulation(
                        sub_population_id=sub_population_id,
                        task=task_obj,
                        evaluation=evaluation_obj,
                        file_metadata=file_metadata_obj
                    )

                    sub_population_obj.save()

                    context.update({
                        "sub_population_obj": sub_population_obj
                    })
                else:
                    sub_population_obj = sub_population_obj[0] if sub_population_obj.exists() else None

                    context.update({
                        "sub_population_obj": sub_population_obj
                    })

                # 5. Explain
                explain_objs = Explain.objects.filter(evaluation_id__exact=evaluation_obj.evaluation_id)

                explain_obj = explain_objs[0] if len(explain_objs) > 0 else None

                if explain_obj is not None:

                    context.update({
                        "explain_obj": explain_obj
                    })

                    if explain_obj.task.status == 2:
                        explain_pdp_objs = ExplainPdp.objects.filter(explain_id__exact=explain_obj.explain_id)

                        form_selected_explain_pdp_feature = request.GET.get("form_selected_explain_pdp_feature",
                                                                            explain_pdp_objs[0].feature)

                        form_selected_explain_pdp_id = request.GET.get("form_selected_explain_pdp_id",
                                                                       explain_pdp_objs[0].explain_pdp_id)
                        context.update({
                            "explain_pdp_objs": explain_pdp_objs,
                            "form_selected_explain_pdp_id": form_selected_explain_pdp_id,
                            "form_selected_explain_pdp_feature": form_selected_explain_pdp_feature,
                        })

                        if evaluation_obj.model.experiment.problem_type == "regression":
                            explain_pdp_regress_objs = ExplainPdpRegress.objects.filter(explain_pdp_id__exact=
                                                                                        form_selected_explain_pdp_id)

                            pdp_values = explain_pdp_regress_objs[0].pdp_values
                            df_pdp = pd.DataFrame(pdp_values)
                            df_pdp.sort_values("group_order").reset_index(drop=True, inplace=True)

                            pdp = {
                                "category": df_pdp.group_name.values.tolist(),
                                "pdp_value": df_pdp.pdp_value.values.tolist(),
                                "num": df_pdp.num.values.tolist()
                            }

                            context.update({
                                "explain_pdp_regress_objs": explain_pdp_regress_objs,
                                "pdp_data": pdp
                            })

                        else:
                            explain_pdp_class_objs = ExplainPdpClass.objects.filter(explain_pdp_id__exact=
                                                                                    form_selected_explain_pdp_id)

                            form_selected_explain_pdp_class_id = request.GET.get("form_selected_explain_pdp_class_id", None)
                            form_selected_explain_pdp_class = request.GET.get("form_selected_explain_pdp_class")

                            if (form_selected_explain_pdp_class_id == "") or (form_selected_explain_pdp_class_id is None):
                                form_selected_explain_pdp_class_id = explain_pdp_class_objs[0].explain_pdp_class_id

                            if form_selected_explain_pdp_class == "":
                                form_selected_explain_pdp_class = explain_pdp_class_objs[0].class_name

                            # TODO
                            explain_pdp_class_values_objs = ExplainPdpClassValues.objects.filter(explain_pdp_class_id__exact=form_selected_explain_pdp_class_id)

                            if explain_pdp_class_values_objs.exists():
                                df_pdp = pd.DataFrame(explain_pdp_class_values_objs[0].pdp_values)
                                df_pdp.sort_values("group_order").reset_index(drop=True, inplace=True)

                                pdp = {
                                    "category": df_pdp.group_name.values.tolist(),
                                    "pdp_value": df_pdp.pdp_value.values.tolist(),
                                    "num": df_pdp.num.values.tolist()
                                }
                                context.update({"pdp_data": pdp})

                            context.update({
                                "explain_pdp_class_objs": explain_pdp_class_objs,
                                "form_selected_explain_pdp_class": form_selected_explain_pdp_class,
                                "form_selected_explain_pdp_class_id": form_selected_explain_pdp_class_id,
                            })

        return render(request, 'model/detail.html', context=context)

    if request.method == "POST":
        # Run evaluation
        submit_evaluation = request.POST.get("submit_evaluation", None)
        submit_explain = request.POST.get("submit_explain", None)
        submit_download = request.POST.get("submit_download", None)
        file_id = request.POST.get("submit_file_id", None)

        if submit_evaluation is not None:
            model_id = submit_evaluation

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

        if submit_download is not None:
            evaluation_id = submit_download

            file_url = get_evaluation_url(evaluation_id)

            try:
                with open(file_url, 'rb') as fh:
                    response = HttpResponse(fh.read(),
                                            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    response['Content-Disposition'] = 'attachment; filename={}.xlsx'.format("evaluation")
                    return response
            except Exception as e:
                mes = '<h1>Can not open excel file at local.  </h1>. ERROR : {}'.format(e)
                return HttpResponse(mes)

        if submit_explain is not None:
            evaluation_id = submit_explain

            evaluation_obj = Evaluation.objects.get(pk=evaluation_id)

            explain_id = "{}".format(evaluation_id)

            task_id = explain_task.delay(explain_id)

            task_obj = Task(
                task_id=task_id,
                status=config.TASK_STATUS.get('PENDING')
            )
            task_obj.save()

            explain_obj = Explain(
                explain_id=explain_id,
                task=task_obj,
                evaluation=evaluation_obj,
            )
            explain_obj.save()

            return redirect("model_detail", pk=evaluation_obj.model_id)


@csrf_exempt
def evaluation_sub_population(request):
    """
    API request from worker
    :param request:
    :return:
    """
    if request.method == "POST":
        body_unicode = request.body.decode('utf-8')
        agrs = json.loads(body_unicode)

        sub_population_id = agrs.get("sub_population_id", None)
        sub_population = agrs.get("sub_population", None)

        if sub_population_id is not None:
            try:
                sub_population_obj = EvaluationSubPopulation.objects.get(pk=sub_population_id)

                sub_population_obj.sub_population = sub_population

                sub_population_obj.save()
            except Exception as e:
                return JsonResponse({'code': 405, 'description': "Fail. Error {}".format(e)})

            return JsonResponse({'code': 200, 'description': "success"})


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
def evaluation_predict_actual(request):
    """
    API used to call from worker
    :param request:
    :return:
    """
    if request.method == "POST":
        body_unicode = request.body.decode('utf-8')
        agrs = json.loads(body_unicode)

        evaluation_id = agrs.get("evaluation_id", None)
        predict_vs_actual = agrs.get("predict_vs_actual", None)

        evaluation_obj = Evaluation.objects.get(pk=evaluation_id)

        eval_predict_actual_obj = EvaluationPredictActual(
            evaluation=evaluation_obj,
            predict_vs_actual=predict_vs_actual
        )
        eval_predict_actual_obj.save()

        return JsonResponse({'code': 200, 'description': "success"})

    return JsonResponse({'code': 403, 'description': "Not support this method"})


@csrf_exempt
def evaluation_class(request):
    """
    API used to call from worker
    :param request:
    :return:
    """
    if request.method == "POST":
        body_unicode = request.body.decode('utf-8')
        agrs = json.loads(body_unicode)

        evaluation_id = agrs.get("evaluation_id", None)
        class_id = agrs.get("class_id")
        class_name = agrs.get("class_name")

        evaluation_obj = Evaluation.objects.get(pk=evaluation_id)

        evaluation_class_obj = EvaluationClass(
            evaluation_class_id=class_id,
            evaluation_class_name=class_name,
            evaluation=evaluation_obj
        )

        evaluation_class_obj.save()

        return JsonResponse({'code': 200, 'description': "success"})

    return JsonResponse({'code': 403, 'description': "Not support this method"})


@csrf_exempt
def evaluation_class_roc_lift(request):
    """
    API used to call from worker
    :param request:
    :return:
    """
    if request.method == "POST":
        body_unicode = request.body.decode('utf-8')
        agrs = json.loads(body_unicode)

        class_id = agrs.get("class_id")
        scores = agrs.get("scores")

        eval_class_obj = EvaluationClass.objects.get(pk=class_id)

        eval_class_roc_lift_obj = EvaluationClassRocLift(
            evaluation_class=eval_class_obj,
            scores=scores
        )
        eval_class_roc_lift_obj.save()

        return JsonResponse({'code': 200, 'description': "success"})

    return JsonResponse({'code': 403, 'description': "Not support this method"})


@csrf_exempt
def evaluation_class_predict_distribution(request):
    """
    API used to call from worker
    :param request:
    :return:
    """
    if request.method == "POST":
        body_unicode = request.body.decode('utf-8')
        agrs = json.loads(body_unicode)

        class_id = agrs.get("class_id", None)
        predict_distribution = agrs.get("predict_distribution", None)

        evaluation_class_obj = EvaluationClass.objects.get(pk=class_id)

        evaluation_class_distribution_obj = EvaluationClassDistribution(
            predict_distribution=predict_distribution,
            evaluation_class=evaluation_class_obj
        )
        evaluation_class_distribution_obj.save()

        return JsonResponse({'code': 200, 'description': "success"})

    return JsonResponse({'code': 403, 'description': "Not support this method"})


@csrf_exempt
def experiment_detail(request, pk):

    # Processing create predict
    if request.POST.get("create_predict"):

        file_id = request.POST.get('select_file', None)

        # Predict data is uploaded
        if file_id is None:
            upload_file = request.FILES.get('upload_file', None)

            if upload_file is None:
                raise Http404("Upload file does not exist")

            # upload file and save to database
            file_id = uuid.uuid1()
            file_path = get_file_url(file_id)
            default_storage.save(file_path, upload_file)

            task_id = upload_file_task.delay(file_id)
            task_id = str(task_id)

            task_obj = Task(
                task_id=task_id,
                status=config.TASK_STATUS.get('PENDING')
            )
            task_obj.save()

            # Save file info
            file_obj = File(
                file_id=file_id, file_name='{}'.format(upload_file),
                is_external=True,
                file_path=file_path, is_delete=False, task=task_obj
            )
            file_obj.save()

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
        predict_id = request.POST.get("delete_id")

        predict_obj = Predict.objects.get(pk=predict_id)
        predict_obj.is_delete = True
        predict_obj.save()

        return redirect("experiment_detail", pk=pk)

    if request.POST.get("delete_deployment"):

        deployment_id = request.POST.get("delete_id")

        deployment_obj = Deployment.objects.get(pk=deployment_id)
        deployment_obj.is_delete = True
        deployment_obj.save()

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

    # Processing deploy
    if request.POST.get("create_deploy"):
        model_id = request.POST.get("deploy_model")
        connection_id = request.POST.get("deploy_connection")
        output_table = request.POST.get("output_table")
        deploy_name = request.POST.get("deploy_name")

        try:
            model_obj = Model.objects.get(pk=model_id)
            connection_obj = Connection.objects.get(pk=connection_id)
        except Exception as e:
            mes = '<h1>Not found predict id.  </h1>. ERROR : {}'.format(e)
            return HttpResponse(mes)

        deploy_id = str(uuid.uuid1())

        # Load predict file
        task_id = deploy_task.delay(deploy_id)

        task_obj = Task(
            task_id=task_id,
            status=config.TASK_STATUS.get('PENDING')
        )
        task_obj.save()

        _current_time = datetime.now().strftime('%Y%m%d_%H%M')
        teradata_model_id = "{}_{}".format(_current_time, model_obj.model_name)

        deploy_obj = Deployment(
            deploy_id=deploy_id,
            deploy_name=deploy_name,
            output_table=output_table,
            model=model_obj,
            task=task_obj,
            experiment_id=pk,
            teradata_model_id=teradata_model_id,
            connection=connection_obj
        )

        deploy_obj.save()

        return redirect("experiment_detail", pk=pk)

    # MAIN.
    if request.method == "GET":
        experiment_obj = Experiment.get_experiment_ui(pk)

        context = {
            "experiment_obj": experiment_obj,
            "feature_importance": None,
            "models_performance": None,
        }

        if experiment_obj.task.status == 2:

            model_objs = Model.objects.filter(experiment_id__exact=pk)

            predict_objs = Predict.get_predict(pk)

            file_objs = File.get_success_files().filter(is_external=False)

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

                _data = {"x": round(model_obj.score_val, 2), "y": round(model_obj.score_test, 2), "z": r, "name": model_obj.model_name}

                models_performance.append(_data)

            # Add infor
            connection_objs = Connection.get_connections()
            deployment_objs = Deployment.get_deployments(pk)

            context = {
                "experiment_obj": experiment_obj,
                "model_objs": model_objs,
                "predict_objs": predict_objs,
                "file_objs": file_objs,
                "best_model_obj": best_model_obj,
                "feature_importance": feature_importance,
                "models_performance": models_performance,
                "connection_objs": connection_objs,
                "deployment_objs": deployment_objs
            }

    return render(request, 'experiment/detail.html', context=context)


@csrf_exempt
def deploy(request):
    """
    API. Used to get explain info.
    :param request:
    :return:
    """
    if request.method == "GET":


        deploy_id = request.GET.get('deploy_id', None)

        deploy_info = Deployment.get_deploy_api(deploy_id)

        return JsonResponse(deploy_info)

    if request.method == "POST":

        body_unicode = request.body.decode('utf-8')
        agrs = json.loads(body_unicode)

        deploy_id = agrs.get('deploy_id', None)
        if deploy_id is None:
            return JsonResponse({'code': 404, 'description': 'Task id must be not none'})

        try:
            deploy_obj = Deployment.objects.get(pk=deploy_id)
            deploy_obj.sql_preprocess = agrs.get('sql_preprocess')
            deploy_obj.sql_prediction = agrs.get('sql_prediction')
            deploy_obj.save()

        except Deployment.DoesNotExist:
            return JsonResponse({'code': 405, 'description': 'Deployment id not found'})

        return JsonResponse({'code': 200, 'description': 'Success'})


def experiment_create(request, pk=None):

    form = {
        "view_id": request.POST.get('view_id', 2 if pk is not None else 1),
        "file_id": request.POST.get('file_id', pk),
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

    file_objs = File.get_files().filter(is_external=False)

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
    file_objs = File.get_files().filter(is_external=False)

    connection_objs = Connection.get_connections()

    task_id_list = [file_obj.task.task_id for file_obj in file_objs]

    task_status_list = [file_obj.task.status for file_obj in file_objs]

    context = {
        'file_objs': file_objs,
        'connection_objs': connection_objs,
        'selected_connection_obj': connection_objs[0] if len(connection_objs) > 0 else None,
        'task_id_list': task_id_list,
        'task_status_list': task_status_list,
        'selected_tab': "files-tab",
    }

    if request.method == 'GET':
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

            # # TODO : Return file metadata
            # df_file = Input(file_path).from_csv()

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

        # 4. Create db connection
        if request.POST.get("create_connection"):
            connection_name = request.POST.get('connection_name', None)
            host = request.POST.get('host', None)
            user_name = request.POST.get('user_name', None)
            password = request.POST.get('password', None)
            database = request.POST.get('database', None)

            connection_obj = Connection(
                host=host,
                connection_name=connection_name,
                user_name=user_name,
                database=database,
                is_delete=False,
                password=Connection.encrypt_password(password)
            )

            connection_obj.save()

        # 5. Selected connection
        if request.POST.get("select_connection"):
            connection_id = request.POST.get('connection_id', None)
            if connection_id:

                selected_connection_obj = Connection.get_connection(connection_id)

                context.update({
                    'selected_connection_obj': selected_connection_obj,
                    'selected_tab': "connect-db-tab",
                })

                return render(request, 'file/index.html', context=context)

        # 6. Delete connection
        if request.POST.get("delete_connection"):
            delete_connection_id = request.POST.get('delete_connection_id', None)

            delete_connection_obj = Connection.objects.get(pk=delete_connection_id)

            delete_connection_obj.is_delete = True
            delete_connection_obj.save()

            connection_objs = Connection.get_connections()

            context.update({
                'selected_tab': "connect-db-tab",
                'connection_objs': connection_objs,
                'selected_connection_obj': connection_objs[0] if len(connection_objs) > 0 else None,
            })
            return render(request, 'file/index.html', context=context)

        # 7. Get detail of connection
        if request.POST.get("get_tables_of_db"):

            connection_id = request.POST.get("connection_id")

            connection_obj = Connection.get_connection(connection_id)

            tables = connection_obj.get_tables_from_db()

            context.update({
                'selected_connection_obj': connection_obj,
                'selected_tab': "connect-db-tab",
                'connection_db': tables
            })

            return render(request, 'file/index.html', context=context)

        # 8. Ingest table
        if request.POST.get("ingest_table"):
            print("DEBUG DANGEROUS! Re-ingest data")
            table_name = request.POST.get("table_name")
            connection_id = request.POST.get("connection_id")

            connection_obj = Connection.get_connection(connection_id, decrypt_password=False)

            file_id = uuid.uuid1()

            #  Send task into Worker
            task_id = ingest_file_from_db_task.delay(file_id,
                                                     connection_obj.host,
                                                     connection_obj.user_name,
                                                     connection_obj.password,
                                                     connection_obj.database,
                                                     table_name,
                                                     )
            task_id = str(task_id)

            task_obj = Task(
                task_id=task_id,
                status=config.TASK_STATUS.get('PENDING')
            )
            task_obj.save()

            # Add file into file obj. Todo : Add two extra information into file, - relation into Connection
            file_obj = File(
                file_id=file_id,
                file_name="{}_{}".format(connection_obj.connection_name, table_name),
                file_path=None,
                is_delete=False,
                task=task_obj
            )
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
def explain(request):
    """
    API. Used to get explain info.
    :param request:
    :return:
    """
    if request.method == "GET":
        explain_id = request.GET.get('explain_id', None)

        return JsonResponse(Explain.get_explain_api(explain_id))


@csrf_exempt
def explain_pdp(request):
    """
    API. Used to update explain pdp
    :param request:
    :return:
    """
    if request.method == "POST":

        body_unicode = request.body.decode('utf-8')
        agrs = json.loads(body_unicode)

        explain_id = agrs.get("explain_id")
        pdp_list = agrs.get("pdp_list")

        explain_obj = Explain.objects.get(pk=explain_id)

        for pdp in pdp_list:
            feature = pdp.get("feature")

            explain_pdp_id = "{}_{}".format(explain_id, str(feature).strip())
            explain_pdp_obj = ExplainPdp(
                explain_pdp_id=explain_pdp_id,
                explain=explain_obj,
                feature=feature
            )
            explain_pdp_obj.save()

            if explain_obj.evaluation.model.experiment.problem_type == "regression":
                explain_pdp_regress_obj = ExplainPdpRegress(
                    explain_pdp=explain_pdp_obj,
                    pdp_values=pdp.get("pdp_values")
                )
                explain_pdp_regress_obj.save()
            else:
                for pdp_class in pdp.get("pdp_values"):

                    class_name = pdp_class.get("class_name")
                    pdp_values = pdp_class.get("pdp_values")

                    explain_pdp_class_obj = ExplainPdpClass(
                        explain_pdp_class_id=str(uuid.uuid1()),
                        explain_pdp=explain_pdp_obj,
                        class_name=class_name,
                    )
                    explain_pdp_class_obj.save()

                    explain_pdp_class_values_obj = ExplainPdpClassValues(
                        explain_pdp_class=explain_pdp_class_obj,
                        pdp_values=pdp_values
                    )
                    explain_pdp_class_values_obj.save()

        return JsonResponse({'code': 200, 'description': 'Success'})


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


def task_api(request):
    """
    API. Used to get task status.
    :param request:
        - task_id : Created when experiment run
        - status id : 'PENDING': 0, 'STARTED': 1, 'SUCCESS': 2, 'FAILURE': 3, 'RETRY': 4, 'REVOKED': 5
        - description
    :return:
    """
    if request.method == "GET":
        task_id = request.GET.get('task_id', None)

        task_obj = Task.objects.get(pk=task_id)

        return JsonResponse({'code': 200, 'description': 'Success', 'result': task_obj.as_json()})
