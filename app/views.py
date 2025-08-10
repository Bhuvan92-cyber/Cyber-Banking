from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import HttpResponse, FileResponse
from .forms import RegisterForm, LoginForm, UploadCSVForm
from .models import UploadedDataset, PreprocessingResult
from sklearn.impute import SimpleImputer
from django.contrib import messages
from django.urls import reverse


from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import hashlib
import io
import base64

import os
import pandas as pd
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from django.views.decorators.csrf import csrf_exempt
from django.utils.html import escape  # Required for safe HTML
import matplotlib
matplotlib.use('Agg')

# Memory stores
DATASET_PATH = ''
PROCESSED_DATA = None
MODELS_ACCURACY = {}
MODELS_RESULTS = {}
#MODELS_ACCURACY['full_report'] = report_text

# Global cache dictionary keyed by dataset hash and algorithm
DATASET_CACHE = {}

# Global state variables (initialize once)

LAST_ANALYZED_HASH = None
BEST_MODEL_OBJECT = None
BEST_MODEL = None
BEST_MODEL_NAME = None





def home(request):
    if request.user.is_authenticated:
        return redirect('dashboard')  # Automatically forward if already logged in
    return render(request, 'base.html')


def register_user(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = RegisterForm()
    return render(request, 'register.html', {'form': form})

def login_user(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            user = authenticate(username=form.cleaned_data['username'],
                                password=form.cleaned_data['password'])
            if user:
                login(request, user)
                if user.is_superuser:
                    return redirect('admin_panel')
                return redirect('dashboard')
            return HttpResponse("Invalid username or password.")
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})

@login_required
def logout_user(request):
    logout(request)
    return redirect('home')

@login_required
def dashboard(request):
    return render(request, 'dashboard_user.html')

@login_required
def admin_panel(request):
    users = User.objects.filter(is_superuser=False)
    return render(request, 'dashboard_admin.html', {'users': users})

@login_required
def delete_user(request, user_id):
    if request.user.is_superuser:
        User.objects.get(id=user_id).delete()
    return redirect('admin_panel')

def compute_dataset_hash(df):
    # Compute a unique hash for the dataset contents (CSV string)
    csv_str = df.to_csv(index=False)
    return hashlib.sha256(csv_str.encode('utf-8')).hexdigest()

@login_required
def upload_dataset(request):
    global DATASET_CACHE

    if request.method == 'POST':
        form = UploadCSVForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['file']
            dataset_dir = os.path.join(settings.MEDIA_ROOT, 'datasets')
            os.makedirs(dataset_dir, exist_ok=True)

            path = os.path.join(dataset_dir, uploaded_file.name)

            # Save uploaded CSV to disk
            with open(path, 'wb+') as dest:
                for chunk in uploaded_file.chunks():
                    dest.write(chunk)

            # Load dataset into DataFrame
            df = pd.read_csv(path)

            # Compute dataset hash
            dataset_hash = compute_dataset_hash(df)

            # ---- Key Change: Always replace dataset in cache ----
            DATASET_CACHE[dataset_hash] = {
                'dataframe': df,
                'results': {}  # Reset results when new dataset uploaded
            }

            # Update session
            request.session['DATASET_HASH'] = dataset_hash

            # Remove any existing UploadedDataset entry for this user with same hash
            UploadedDataset.objects.filter(user=request.user, file__icontains=uploaded_file.name).delete()

            # Create new record
            UploadedDataset.objects.create(user=request.user, file=uploaded_file)

            messages.success(request, "Dataset uploaded successfully.")

            return render(request, 'upload_dataset.html', {'form': form})
        else:
            messages.error(request, "There was an error uploading the file.")
    else:
        form = UploadCSVForm()

    return render(request, 'upload_dataset.html', {'form': form})



@login_required
def preprocess_dataset(request):
    global DATASET_CACHE, PROCESSED_DATA
    # Retrieve dataset hash from session
    dataset_hash = request.session.get('DATASET_HASH')
    
    if not dataset_hash or dataset_hash not in DATASET_CACHE:
        return render(request, 'preprocess_result.html', {
            'error': "No dataset found or dataset not processed."
        })

    # Retrieve the DataFrame from the cache
    df = DATASET_CACHE[dataset_hash]['dataframe']
    original_rows = df.shape[0]

    # Drop duplicates
    df.drop_duplicates(inplace=True)
    rows_after = df.shape[0]

    # Update the cached DataFrame
    DATASET_CACHE[dataset_hash]['dataframe'] = df

    # Store preprocessing results in the database
    PreprocessingResult.objects.create(
        user=request.user,
        original_rows=original_rows,
        rows_after_cleaning=rows_after
    )
    PROCESSED_DATA = df.copy()
    context = {
        'original_rows': original_rows,
        'rows_after': rows_after
    }
    return render(request, 'preprocess_result.html', context)


@login_required
def run_algorithm(request, algo):
    global DATASET_CACHE, PROCESSED_DATA, MODELS_ACCURACY

    dataset_hash = request.session.get('DATASET_HASH')
    if not dataset_hash or dataset_hash not in DATASET_CACHE:
        return HttpResponse("<h3>No dataset uploaded or dataset not found in cache.</h3>")

    # Check if result is already cached
    cached_results = DATASET_CACHE[dataset_hash]['results'].get(algo)
    if cached_results:
        return HttpResponse(f"""
            <h3>{algo.upper()} Results (Cached):</h3>
            <div style="border: 1px solid #ccc; padding: 10px; margin: 10px 0;">
                <p><strong>Accuracy:</strong> {cached_results['accuracy']*100:.2f}%</p>
                <h4>Classification Report (Visual):</h4>
                <img src="data:image/png;base64,{cached_results['report_img']}" style="max-width:100%; border:1px solid #ccc;"/>
                <h4>Confusion Matrix (Heatmap):</h4>
                <img src="data:image/png;base64,{cached_results['matrix_img']}" style="max-width:100%; border:1px solid #ccc;"/>
                <h4>Raw Classification Report:</h4>
                <pre>{cached_results['report']}</pre>
                <h4>Raw Confusion Matrix:</h4>
                <pre>{cached_results['matrix']}</pre>
            </div>
            <a href="{reverse('dashboard')}" style="padding: 10px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px;">Back to Dashboard</a>
        """)

    df = DATASET_CACHE[dataset_hash]['dataframe'].copy()

    # Drop irrelevant columns if present
    df.drop(columns=[col for col in ['id', 'name'] if col in df.columns], inplace=True, errors='ignore')

    X = df.drop('target', axis=1)
    y = df['target']

    label_encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    trained_columns = X.columns.tolist()
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    models = {
        'rf': RandomForestClassifier(random_state=42),
        'gb': GradientBoostingClassifier(random_state=42),
        'svm': SVC(probability=True, random_state=42),
        'lr': LogisticRegression(max_iter=1000, random_state=42)
    }

    accuracy_results = {}
    model_dir = os.path.join('models')
    os.makedirs(model_dir, exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracy_results[name] = acc
        joblib.dump(model, os.path.join(model_dir, f'{name}_model.pkl'))

    best_model_name = max(accuracy_results, key=accuracy_results.get)
    request.session['BEST_MODEL'] = best_model_name

    joblib.dump(label_encoders, os.path.join(model_dir, 'label_encoders.pkl'))
    joblib.dump(imputer, os.path.join(model_dir, 'imputer.pkl'))
    joblib.dump(trained_columns, os.path.join(model_dir, 'trained_columns.pkl'))

    request.session['MODELS_ACCURACY'] = accuracy_results
    MODELS_ACCURACY = accuracy_results
    PROCESSED_DATA = df.copy()

    # Use selected model to predict and report
    selected_model = models.get(algo)
    if selected_model is None:
        return HttpResponse("<h3>Invalid algorithm specified.</h3>")

    selected_model.fit(X_train, y_train)
    y_pred = selected_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)

    # Visuals
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    classes = list(report.keys())[:-3]
    precision = [report[cls]['precision'] for cls in classes]
    recall = [report[cls]['recall'] for cls in classes]
    f1 = [report[cls]['f1-score'] for cls in classes]

    x = range(len(classes))
    ax1.bar([i - 0.2 for i in x], precision, width=0.2, label='Precision', color='skyblue')
    ax1.bar(x, recall, width=0.2, label='Recall', color='lightgreen')
    ax1.bar([i + 0.2 for i in x], f1, width=0.2, label='F1-Score', color='salmon')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.set_ylim([0, 1])
    ax1.set_title('Classification Report')
    ax1.legend()
    plt.tight_layout()

    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png')
    buf1.seek(0)
    report_img_base64 = base64.b64encode(buf1.read()).decode('utf-8')
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(4, 3))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    plt.tight_layout()

    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png')
    buf2.seek(0)
    matrix_img_base64 = base64.b64encode(buf2.read()).decode('utf-8')
    plt.close(fig2)

    # Cache result
    DATASET_CACHE[dataset_hash]['results'][algo] = {
        'accuracy': acc,
        'report': classification_report(y_test, y_pred),
        'matrix': matrix.tolist(),
        'report_img': report_img_base64,
        'matrix_img': matrix_img_base64
    }

    return HttpResponse(f"""
        <h3>{algo.upper()} Results:</h3>
        <div style="border: 1px solid #ccc; padding: 10px; margin: 10px 0;">
            <p><strong>Accuracy:</strong> {acc*100:.2f}%</p>
            <h4>Classification Report (Visual):</h4>
            <img src="data:image/png;base64,{report_img_base64}" style="max-width:100%; border:1px solid #ccc;"/>
            <h4>Confusion Matrix (Heatmap):</h4>
            <img src="data:image/png;base64,{matrix_img_base64}" style="max-width:100%; border:1px solid #ccc;"/>
            <h4>Raw Classification Report:</h4>
            <pre>{classification_report(y_test, y_pred)}</pre>
            <h4>Raw Confusion Matrix:</h4>
            <pre>{matrix}</pre>
        </div>
        <a href="{reverse('run_ml_algorithms')}" style="padding: 10px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px;"> Back to Run ML Algorithms</a>
    """)


def run_ml_algorithms(request):
    return render(request, 'run_ml_algorithms.html')


@login_required
def generate_report(request):
    all_algos = {
        'rf': 'Random Forest',
        'gb': 'Gradient Boosting',
        'svm': 'SVM',
        'lr': 'Logistic Regression'
    }


    # Assuming MODELS_RESULTS is defined somewhere in your code
    MODELS_RESULTS = request.session.get('MODELS_ACCURACY', {})


    # Prepare data for charting
    names = []
    scores = []
    for key, label in all_algos.items():
        score = MODELS_RESULTS.get(key)
        if isinstance(score, (int, float)) and not np.isnan(score) and score > 0:
            names.append(label)
            scores.append(score)


    if not scores:
        return HttpResponse("<h3>No model has been run yet. Please run at least one algorithm first.</h3>")



    # Pie Chart
    plt.figure(figsize=(6, 6))
    plt.pie(scores, labels=names, autopct='%1.1f%%', startangle=90)
    plt.title("Algorithm Accuracy Comparison - Pie Chart")
    pie_path = os.path.join(settings.MEDIA_ROOT, 'pie.png')
    plt.savefig(pie_path)
    plt.close()  # Prevent memory issues


    # Line Chart
    plt.figure()
    plt.plot(names, scores, marker='o', linestyle='-', color='blue')
    plt.title("Algorithm Accuracy - Line Chart")
    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    line_path = os.path.join(settings.MEDIA_ROOT, 'line.png')
    plt.savefig(line_path)
    plt.close()


    return HttpResponse(f"""
        <h3>Graphical Report</h3>
        <img src="/media/pie.png" width="400" alt="Pie Chart"/>
        <img src="/media/line.png" width="400" alt="Line Chart"/>
        <br><br>
        <a href="{reverse('dashboard')}" style="padding: 10px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px;">Back to Dashboard</a>
    """)


@login_required
def predict_best_algorithm(request):
    global DATASET_CACHE

    dataset_hash = request.session.get('DATASET_HASH')
    if not dataset_hash or dataset_hash not in DATASET_CACHE:
        return HttpResponse("Please upload and run analysis on a dataset first.")

    results = DATASET_CACHE[dataset_hash].get('results', {})
    if not results:
        return HttpResponse("No algorithm results found. Please run the algorithms first.")

    # Find the algorithm with the highest accuracy
    best_algo, best_result = max(results.items(), key=lambda x: x[1]['accuracy'])

    # Extract details
    accuracy = best_result['accuracy'] * 100  # percentage
    report = best_result.get('report', '')
    matrix = best_result.get('matrix', '')
    y_true = best_result.get('y_true', [])
    y_pred = best_result.get('y_pred', [])

    # === 1. Generate Confusion Matrix Heatmap ===
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"Confusion Matrix - {best_algo.upper()}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    image_uri = f"data:image/png;base64,{image_base64}"
    
    # === 2. Generate Classification Report Heatmap ===
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose().round(2)
    report_df = report_df.drop(columns=['support'], errors='ignore')

    fig2, ax2 = plt.subplots(figsize=(6, len(report_df) * 0.5 + 1))
    sns.heatmap(
        report_df, 
        annot=True, 
        cmap="YlGnBu", 
        fmt=".2f", 
        linewidths=0.5, 
        ax=ax2,
        vmin=0, vmax=1  # Ensure scale is correct
    )
    ax2.set_title(f"Classification Report - {best_algo.upper()}")
    plt.yticks(rotation=0)


    buf2 = io.BytesIO()
    plt.tight_layout()
    fig2.savefig(buf2, format='png')
    plt.close(fig2)
    buf2.seek(0)
    report_image_base64 = base64.b64encode(buf2.getvalue()).decode('utf-8')
    report_image_uri = f"data:image/png;base64,{report_image_base64}"

    escaped_report = escape(report)
    escaped_matrix = escape(str(matrix))

    html = f"""
    <h2>Best Algorithm: {best_algo.upper()}</h2>
    <div style="border: 1px solid #ccc; padding: 10px; margin: 10px 0;">
        <p><strong>Accuracy:</strong> {accuracy:.2f}%</p>
        <h3>Classification Report:</h3>
        <pre>{escaped_report}</pre>
        <h3>Classification Report (Visual):</h3>
        <img src="{report_image_uri}" alt="Classification Report Heatmap" style="max-width: 100%; height: auto; border: 1px solid #888;"/>
        <h3>Confusion Matrix:</h3>
        <img src="{image_uri}" alt="Confusion Matrix" style="max-width: 100%; height: auto; border: 1px solid #888;"/>
    </div>
    <a href="{reverse('dashboard')}" style="padding: 10px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px;">Back to Dashboard</a>
    <a href="javascript:void(0);" onclick="copyToClipboard()" style="padding: 10px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px;">Copy Result</a>
    <a href="{reverse('download_report')}" style="padding: 10px; background-color: #007BFF; color: white; text-decoration: none; border-radius: 5px;">Download Report</a>
    <script>
        function copyToClipboard() {{
            const resultText = `Best Algorithm: {best_algo.upper()}\\nAccuracy: {accuracy:.2f}%\\nClassification Report: {escaped_report}\\nConfusion Matrix: {escaped_matrix}`;
            navigator.clipboard.writeText(resultText).then(() => {{
                alert('Result copied to clipboard!');
            }}, (err) => {{
                console.error('Could not copy text: ', err);
            }});
        }}
    </script>    
    """
    return HttpResponse(html)


from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io

@login_required
def download_report(request):
    dataset_hash = request.session.get('DATASET_HASH')
    if not dataset_hash or dataset_hash not in DATASET_CACHE:
        return HttpResponse("No report available for download.")

    results = DATASET_CACHE[dataset_hash].get('results', {})
    if not results:
        return HttpResponse("No algorithm results found.")

    # Get best algorithm result
    best_algo, best_result = max(results.items(), key=lambda x: x[1]['accuracy'])
    accuracy = best_result['accuracy'] * 100
    report = best_result.get('report', '')
    matrix = best_result.get('matrix', '')
    y_true = best_result.get('y_true', [])
    y_pred = best_result.get('y_pred', [])

    # Generate confusion matrix image
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"Confusion Matrix - {best_algo.upper()}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    matrix_image = ImageReader(buf)

    # Generate classification report image
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose().round(2)
    report_df = report_df.drop(columns=['support'], errors='ignore')
    fig2, ax2 = plt.subplots(figsize=(6, len(report_df) * 0.5 + 1))
    sns.heatmap(report_df, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5, ax=ax2, vmin=0, vmax=1)
    ax2.set_title(f"Classification Report - {best_algo.upper()}")
    plt.yticks(rotation=0)
    buf2 = io.BytesIO()
    plt.tight_layout()
    fig2.savefig(buf2, format='png')
    plt.close(fig2)
    buf2.seek(0)
    report_image = ImageReader(buf2)

    # Create PDF in memory
    pdf_buffer = io.BytesIO()
    p = canvas.Canvas(pdf_buffer, pagesize=letter)
    width, height = letter

    # Text content
    p.setFont("Helvetica-Bold", 14)
    p.drawString(30, height - 40, f"Best Algorithm: {best_algo.upper()}")
    p.setFont("Helvetica", 12)
    p.drawString(30, height - 60, f"Accuracy: {accuracy:.2f}%")
    
    p.drawString(30, height - 90, "Classification Report (Text):")
    text_object = p.beginText(30, height - 110)
    text_object.setFont("Helvetica", 10)
    for line in report.splitlines():
        text_object.textLine(line)
    p.drawText(text_object)

    # Add Confusion Matrix Image
    p.showPage()
    p.setFont("Helvetica-Bold", 14)
    p.drawString(30, height - 40, "Confusion Matrix")
    p.drawImage(matrix_image, 50, 200, width=500, preserveAspectRatio=True, mask='auto')

    # Add Classification Report Heatmap Image
    p.showPage()
    p.setFont("Helvetica-Bold", 14)
    p.drawString(30, height - 40, "Classification Report Heatmap")
    p.drawImage(report_image, 50, 200, width=500, preserveAspectRatio=True, mask='auto')

    # Finalize PDF
    p.save()
    pdf_buffer.seek(0)

    return FileResponse(pdf_buffer, as_attachment=True, filename='analysis_report.pdf')




@login_required
@csrf_exempt
def predict(request):
    global PROCESSED_DATA, MODELS_ACCURACY, DATASET_CACHE

    if PROCESSED_DATA is None:
        return HttpResponse("<h3>Error: No processed data available for predictions.</h3>")

    if request.method == 'POST':
        try:
            # Step 1: Get input from form
            input_dict = {
                'age': int(request.POST['age']),
                'gender': request.POST['gender'],
                'account_type': request.POST['account_type'],
                'balance': float(request.POST['balance']),
                'kyc_status': request.POST['kyc_status'],
                'credit_score': int(request.POST['credit_score']),
                'transaction_count': int(request.POST['transaction_count']),
                'device_used': request.POST['device_used']
            }
            input_data = pd.DataFrame([input_dict])

            # Step 2: Load pre-trained model and encoders
            best_algo_key = request.session.get('BEST_MODEL')
            if not best_algo_key:
                return HttpResponse("<h3>Error: No model key found in session.</h3>")

            model_path = os.path.join(settings.BASE_DIR, 'models', f'{best_algo_key}_model.pkl')
            encoders_path = os.path.join(settings.BASE_DIR, 'models', 'label_encoders.pkl')
            imputer_path = os.path.join(settings.BASE_DIR, 'models', 'imputer.pkl')
            columns_path = os.path.join(settings.BASE_DIR, 'models', 'trained_columns.pkl')

            if not os.path.exists(model_path):
                return HttpResponse("<h3>Error: Trained model file not found.</h3>")

            model = joblib.load(model_path)
            label_encoders = joblib.load(encoders_path)
            imputer = joblib.load(imputer_path)
            trained_columns = joblib.load(columns_path)

            # Step 3: Encode input data
            for col in input_data.columns:
                if input_data[col].dtype == 'object':
                    if col in label_encoders:
                        le = label_encoders[col]
                        val = input_data[col].iloc[0]
                        if val in le.classes_:
                            input_data[col] = le.transform([val])
                        else:
                            input_data[col] = le.transform([le.classes_[0]])
                    else:
                        input_data[col] = 0  # fallback

            # Step 4: Align input with training columns
            for col in trained_columns:
                if col not in input_data.columns:
                    input_data[col] = 0  # Add missing columns as 0

            input_data = input_data[trained_columns]  # Reorder columns

            # Step 5: Impute if needed
            input_imputed = pd.DataFrame(imputer.transform(input_data), columns=trained_columns)

            # Step 6: Predict
            prediction_label = model.predict(input_imputed)[0]

            # Optional: Feature importance
            graph = ""
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(10, 5))
                sns.barplot(x=trained_columns, y=model.feature_importances_)
                plt.xticks(rotation=45)
                plt.title("Feature Importance")
                plt.tight_layout()

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                graph = base64.b64encode(buffer.read()).decode()
                buffer.close()
                graph = f"<img src='data:image/png;base64,{graph}' class='img-fluid'>"
                plt.close()

            return HttpResponse(f"""
                <h3>Predicted result: <span style='color:blue'>{prediction_label}</span></h3>
                {graph}
                <br>
                <a href="{reverse('dashboard')}" style="padding: 10px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px;">Back to Dashboard</a>
            """)

        except Exception as e:
            return HttpResponse(f"<h3>Error during prediction: {str(e)}</h3>")

    return render(request, 'prediction.html')
