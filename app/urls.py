from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register_user, name='register'),
    path('login/', views.login_user, name='login'),
    path('logout/', views.logout_user, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('admin-panel/', views.admin_panel, name='admin_panel'),
    path('delete-user/<int:user_id>/', views.delete_user, name='delete_user'),
    
    # ML Operations
    path('upload-dataset/', views.upload_dataset, name='upload_dataset'),
    path('preprocess/', views.preprocess_dataset, name='preprocess_dataset'),
    path('run-ml-algorithms/', views.run_ml_algorithms, name='run_ml_algorithms'),
    path('run-algorithm/<str:algo>/', views.run_algorithm, name='run_algorithm'),
    path('generate-report/', views.generate_report, name='generate_report'),
    path('predict/', views.predict, name='predict'),
    path('predict-best/', views.predict_best_algorithm, name='predict_best_algorithm'),
    path('download_report', views.download_report, name='download_report'),
]
 
