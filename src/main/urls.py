from django.urls import path
from main import views

urlpatterns = [
    path('', views.index, name='index'),
    path('experiment', views.experiment, name='experiment'),

    path('model', views.model, name='model'),
    path('model/<str:pk>', views.model_detail, name='model_detail'),

    path('predict', views.predict, name='predict'),

    path('evaluation', views.evaluation, name='evaluation'),

    path('evaluation/sub_population', views.evaluation_sub_population, name='evaluation_sub_population'),
    path('evaluation/evaluation_class', views.evaluation_class, name='evaluation_class'),
    path('evaluation/evaluation_predict_actual', views.evaluation_predict_actual, name='evaluation_predict_actual'),
    path('evaluation/evaluation_class/roc_lift', views.evaluation_class_roc_lift, name='evaluation_class_roc_lift'),
    path('evaluation/evaluation_class/distribution', views.evaluation_class_predict_distribution,
         name='evaluation_class_predict_distribution'),

    path('explain', views.explain, name='explain'),
    path('explain/pdp', views.explain_pdp, name='explain_pdp'),
    # path('explain/pdp/regress', views.explain_pdp_regress, name='explain_pdp_regress'),
    # path('explain/pdp/class', views.explain_pdp_class, name='explain_pdp_class'),
    # path('explain/pdp/class/values', views.explain_pdp_class_values, name='explain_pdp_class_values'),

    path('experiment/create/', views.experiment_create, name='experiment_create'),
    path('experiment/create/<str:pk>', views.experiment_create, name='experiment_create'),

    path('experiment/detail/<str:pk>', views.experiment_detail, name='experiment_detail'),

    path('file/index/', views.file, name='file'),
    path('file/metadata/', views.file_metadata, name='file_metadata'),
    path('file/eda/<str:pk>', views.file_eda, name='file_eda'),

    path(r'task/', views.task, name='task'),
    path(r'task/<str:pk>', views.task, name='task_detail'),

    path(r'api/v1/task/', views.task_api, name='task_api'),

    path('deploy', views.deploy, name='deploy'),
]
