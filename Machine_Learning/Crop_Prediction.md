
# Crop Recommendation System

## Project Overview
This project develops a machine learning-based crop recommendation system. The goal is to recommend the most suitable crop to cultivate based on various environmental parameters such as Nitrogen (N), Phosphorus (P), Potassium (K) levels in the soil, temperature, humidity, pH, and rainfall. The system utilizes several classification algorithms, with the Random Forest Classifier demonstrating the highest accuracy.

## Dataset
The dataset used is `Crop_recommendation.csv`, which contains various environmental factors and the corresponding optimal crop label. Each entry in the dataset represents a set of conditions and the crop best suited for those conditions.

## Key Steps and Analysis

### 1. Data Loading and Initial Inspection
- The dataset was loaded into a pandas DataFrame.
- Basic information (`df.info()`) and duplicate checks (`df.duplicated().sum()`) were performed, revealing no missing values or duplicates.

### 2. Exploratory Data Analysis (EDA)
- **Class Distribution**: The `label` column shows 22 unique crop types, each with 100 entries, indicating a balanced dataset.
- **Descriptive Statistics**: `df.describe()` provided statistical summaries of numerical features.
- **Box Plots**: Visualizations for Nitrogen, Phosphorus, Potassium, pH, Temperature, Humidity, and Rainfall were generated to identify potential outliers and understand their distributions.
- **Correlation Matrix**: A heatmap was created to visualize correlations between numerical features. `P` and `K` showed the strongest positive correlation.
- **Feature-Label Box Plots**: Box plots of each numerical feature against the crop label were generated to observe feature distributions across different crop types.

### 3. Data Preprocessing
- **Label Encoding**: The categorical `label` column was mapped to numerical values for model training.
- **Feature-Target Split**: The dataset was divided into features (X) and the target variable (y).
- **Feature Scaling**: `MinMaxScaler` was applied to scale the features to a common range, which helps improve the performance of some machine learning algorithms.
- **Train-Test Split**: The data was split into training (70%) and testing (30%) sets to evaluate model performance on unseen data.

### 4. Model Training and Evaluation
Several classification models were trained and evaluated:
- **KNeighborsClassifier (KNN)**
- **Support Vector Classifier (SVC)**
- **DecisionTreeClassifier**
- **RandomForestClassifier**
- **XGBClassifier**

Each model was evaluated using `classification_report`, `accuracy_score`, `precision_score`, `recall_score`, and `f1_score`.

| Model                  | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) |
|------------------------|--------------|---------------|------------|--------------|
| KNeighborsClassifier   | 98.18        | 98.35         | 98.18      | 98.19        |
| SVC                    | 98.79        | 98.81         | 98.79      | 98.79        |
| DecisionTreeClassifier | 98.03        | 98.04         | 98.03      | 98.01        |
| RandomForestClassifier | 99.55        | 99.55         | 99.55      | 99.55        |
| XGBClassifier          | 99.09        | 99.12         | 99.09      | 99.08        |

### 5. Cross-Validation and Hyperparameter Tuning
- **Cross-Validation**: 5-fold cross-validation was performed to ensure model robustness.
  - Decision Tree: Mean CV Accuracy = 98.25%
  - Random Forest: Mean CV Accuracy = 99.48%
  - KNN: Mean CV Accuracy = 97.73%
  - SVC: Mean CV Accuracy = 98.05%
  - XGBoost: Mean CV Accuracy = 98.83%
- **Grid Search**: Hyperparameter tuning was conducted using GridSearchCV for RandomForestClassifier and SVC to find the optimal parameters.
  - **Best RF Params**: `{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}` with Best RF CV Accuracy: `99.55%`
  - **Best SVC Params**: `{'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}` with Best SVC CV Accuracy: `98.64%`

### 6. Best Model Selection and Confusion Matrix
The Random Forest Classifier, with its tuned hyperparameters, achieved the highest test accuracy of **99.55%**. A confusion matrix was generated for the Random Forest model to visualize its performance across all crop classes.

### 7. Model and Scaler Saving
The trained Random Forest model and the MinMaxScaler were saved using `pickle` for future use in the web application.

## Streamlit Web Application
A Streamlit application (`app.py`) was developed to provide an interactive interface for crop recommendation. Users can input the environmental parameters, and the app will predict the most suitable crop.

### Features:
- User-friendly interface for inputting N, P, K, temperature, humidity, pH, and rainfall.
- Utilizes the trained Random Forest model and scaler to make predictions.
- Displays the recommended crop name.

## Technologies Used
- Python
- Pandas (for data manipulation)
- Seaborn & Matplotlib (for data visualization)
- Scikit-learn (for machine learning models, scaling, and evaluation)
- XGBoost (for extreme gradient boosting)
- Streamlit (for web application development)
- Pyngrok (for creating public URLs for the Streamlit app)

## How to Run the Streamlit App (in Google Colab)
1.  **Install Libraries**: Ensure `streamlit` and `pyngrok` are installed.
2.  **Save Model and Scaler**: Run the cells to train the model and save `random_forest_model.sav` and `minmaxscale.sav`.
3.  **Create `app.py`**: Run the `%%writefile app.py` cell to create the Streamlit application file.
4.  **Ngrok Authtoken**: Add your `NGROK_AUTH_TOKEN` to Colab secrets.
5.  **Launch App**: Run the cell that starts Streamlit and connects to ngrok. A public URL will be provided to access the application.
