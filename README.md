# CyberPhysicalBanking

A Django-based web application for cyber-physical banking, featuring user authentication, dataset upload, machine learning model execution, and report generation. This project integrates data science and web development to provide a platform for banking data analysis and prediction.

## Features
- User registration, login, and dashboard (admin and user views)
- Dataset upload and preprocessing
- Run multiple ML algorithms (Random Forest, Gradient Boosting, SVM, Logistic Regression)
- Visualizations (line and pie charts)
- Prediction and best model selection
- PDF report generation
- Admin interface for user and data management

## Project Structure
```
CyberPhysicalBanking/
│
├── app/                  # Django app with models, views, forms, templates
│   ├── migrations/       # Database migrations
│   ├── static/           # Static files (css, js, images)
│   ├── templates/        # HTML templates
│   └── ...
│
├── cyber_physical_banking/ # Django project settings and URLs
│
├── ml_models/            # ML utilities and runner scripts
│
├── models/               # Pre-trained ML model files (pkl)
│
├── media/                # Uploaded datasets and generated images
│
├── db.sqlite3            # SQLite database
├── manage.py             # Django management script
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Requirements
- Python 3.8+
- pip
- (Recommended) Virtual environment (venv)

### Python Packages
All required packages are listed in `requirements.txt`:
- django
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- imbalanced-learn
- pillow
- reportlab

### pip list
Package             Version
------------------- -----------
asgiref             3.8.1
chardet             5.2.0
contourpy           1.3.2
cycler              0.12.1
Django              5.2.1
djangorestframework 3.16.0
fonttools           4.58.0
imbalanced-learn    0.13.0
joblib              1.5.1
kiwisolver          1.4.8
matplotlib          3.10.3
numpy               2.2.6
packaging           25.0
pandas              2.2.3
pillow              11.2.1
pip                 25.1.1
pyparsing           3.2.3
python-dateutil     2.9.0.post0
pytz                2025.2
reportlab           4.4.1
scikit-learn        1.6.1
scipy               1.15.3
seaborn             0.13.2
six                 1.17.0
sklearn-compat      0.1.3
sqlparse            0.5.3
threadpoolctl       3.6.0
tzdata              2025.2
xgboost             3.0.2


## Setup Instructions

### 1. Clone the Repository
```powershell
git clone <repository-url>
cd CyberPhysicalBanking
```

### 2. Create and Activate Virtual Environment
```powershell
python -m venv cyber
.\cyber\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Apply Migrations
```powershell
python manage.py makemigrations
python manage.py migrate
```

### 5. Create Superuser (Admin)
```powershell
python manage.py createsuperuser
# Follow the prompts for username, email, and password
```

### 6. Run the Development Server
```powershell
python manage.py runserver
```
Visit [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.

## Usage Guide

### User Registration & Login
- Register a new user at `/register/`
- Login at `/login/`
- Access user dashboard at `/dashboard/`

### Dataset Upload & Preprocessing
- Upload datasets at `/upload-dataset/`
- Preprocess data at `/preprocess/`

### Machine Learning Algorithms
- Run ML algorithms at `/run-ml-algorithms/`
- View results and visualizations
- Predict using the best model at `/predict-best/`
- Make custom predictions at `/predict/`

### Reports
- Generate PDF reports at `/generate-report/`
- Download visualizations from `/media/`

### Admin Panel
- Access at `/admin/` (login as superuser)
- Manage users, datasets, and results

## File/Folder Descriptions
- `app/` - Main Django app (models, views, forms, templates)
- `ml_models/` - ML logic and utilities
- `models/` - Pre-trained ML model files
- `media/` - Uploaded files and generated images
- `cyber_physical_banking/` - Project settings and URLs
- `requirements.txt` - Python dependencies
- `manage.py` - Django management script

## Notes
- Ensure all dependencies are installed in the virtual environment.
- For any issues with static files, run:
  ```powershell
  python manage.py collectstatic
  ```
- For troubleshooting, check the Django server logs in the terminal.

## License
This project is for educational and demonstration purposes. Please check with the project owner for licensing details.

## Contact
For questions or support, contact: chbhuvansri00@gmail.com
