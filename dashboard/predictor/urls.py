from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('test/', views.test_view, name='test'),
    path('logs/', views.logs_view, name='logs'),
    path('info/', views.model_info_view, name='model_info'),

    # API endpoints
    path('api/predict/', views.predict_api, name='predict_api'),
    path('api/reset/', views.reset_api, name='reset_api'),
    path('api/generate_preset/', views.generate_preset_api, name='generate_preset_api'),
    path('api/log/', views.get_log, name='get_log'),
    path('api/read_log/', views.read_log_ajax, name='read_log_ajax'),
    path('api/run_preset/', views.run_preset_api, name='run_preset_api'),
]
