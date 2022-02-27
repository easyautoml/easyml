from django.urls import path
from main import views

urlpatterns = [
    path('', views.index, name='index'),
    path('experiment', views.experiment, name='experiment'),
    path('model', views.model, name='model'),
    path('model/<str:pk>', views.model_detail, name='model_detail'),
    path('predict', views.predict, name='predict'),
    path('evaluation', views.evaluation, name='evaluation'),
    path('experiment/create/', views.experiment_create, name='experiment_create'),
    path('experiment/detail/<str:pk>', views.experiment_detail, name='experiment_detail'),
    path('file/index/', views.file, name='file'),
    path('file/metadata/', views.file_metadata, name='file_metadata'),
    path('file/eda/<str:pk>', views.file_eda, name='file_eda'),
    path(r'task/', views.task, name='task'),
    path(r'task/<str:pk>', views.task, name='task_detail'),
]
