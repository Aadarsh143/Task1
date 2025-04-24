import numpy as np
import pandas as pd

import kagglehub

import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 800)

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def print_as_pd(array, column_names=None):
    df = pd.DataFrame(array, columns=column_names)
    return df

path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")

print("Path to dataset files:", path)

df_pd = pd.read_csv(path+"/healthcare-dataset-stroke-data.csv")
col_names = df_pd.columns

df = df_pd.to_numpy()


print_as_pd(df)
# Remove first column i.e. id

if(df[0,0] == 'id') :
    df = np.delete(df,0,axis=1)
print_as_pd(df)
headers = df[0]

numeric_data = df[1:].astype(object)

num_cols = [i for i in range(numeric_data.shape[1]) if np.issubdtype(type(numeric_data[0, i]), np.number)]

for col in num_cols:
    col_values = numeric_data[:, col].astype(float)
    col_mean = np.round(np.nanmean(col_values),2)
    col_values[np.isnan(col_values)] = col_mean
    numeric_data[:, col] = col_values

df = np.vstack([headers, numeric_data])

print_as_pd(df)

columns_to_normalize = ['age', 'avg_glucose_level', 'bmi']
col_indices = {col: np.where(df[0] == col)[0][0] for col in columns_to_normalize}


numeric_data = df[1:, list(col_indices.values())].astype(float)


min_vals = np.min(numeric_data, axis=0)
max_vals = np.max(numeric_data, axis=0)
normalized_data = (numeric_data - min_vals) / (max_vals - min_vals)
normalized_data = np.round(normalized_data,4)

for i, col in enumerate(col_indices.values()):
    df[1:, col] = normalized_data[:, i]

print_as_pd(df)

categorical_columns = ['gender', 'ever_married', 'Residence_type']

column_indices = {col: np.where(df[0] == col)[0][0] for col in categorical_columns}

for col, index in column_indices.items():
    unique_values, encoded_values = np.unique(df[1:, index], return_inverse=True)
    df[1:, index] = encoded_values

print_as_pd(df)

work_type_col_idx = np.where(df[0] == 'work_type')[0][0]

df[1:, work_type_col_idx] = np.where(df[1:, work_type_col_idx] == 'children', 'Never_worked', df[1:, work_type_col_idx])

print_as_pd(df)

columns_to_encode = ['work_type', 'smoking_status']
column_indices = {col: np.where(df[0] == col)[0][0] for col in columns_to_encode}

def one_hot_encode(data, col_index):
    unique_values = np.unique(data[1:, col_index])
    one_hot_matrix = np.zeros((data.shape[0] - 1, len(unique_values)), dtype=int)


    for i, val in enumerate(data[1:, col_index]):
        one_hot_matrix[i, np.where(unique_values == val)[0][0]] = 1

    new_headers = [f"{data[0, col_index]}_{val}" for val in unique_values]

    return new_headers, one_hot_matrix

work_type_headers, work_type_encoded = one_hot_encode(df, column_indices['work_type'])
smoking_headers, smoking_encoded = one_hot_encode(df, column_indices['smoking_status'])


new_headers = np.concatenate((df[0], work_type_headers, smoking_headers))
new_data = np.hstack((df[1:], work_type_encoded, smoking_encoded))

df = np.vstack((new_headers, new_data))

print_as_pd(df)

columns_to_remove = ['work_type', 'Residence_type','smoking_status']
column_indices_to_remove = [np.where(df[0] == col)[0][0] for col in columns_to_remove]

df = np.delete(df, column_indices_to_remove, axis=1)

stroke_index = np.where(df[0] == 'stroke')[0][0]

df = np.concatenate((df[:, :stroke_index], df[:, stroke_index + 1:], df[:, stroke_index:stroke_index + 1]), axis=1)

print_as_pd(df)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




# Fill missing values in BMI with the mean
df["bmi"].fillna(df["bmi"].mean(), inplace=True)

# Convert relevant columns to numeric types
df[['age', 'avg_glucose_level', 'bmi', 'stroke']] = df[['age', 'avg_glucose_level', 'bmi', 'stroke']].astype(float)


# 1. Histograms for Numerical Features
import matplotlib.pyplot as plt

numerical_cols = ['age', 'avg_glucose_level', 'bmi']
df[numerical_cols].hist(figsize=(12, 6), bins=30, color='skyblue', edgecolor='black')
plt.suptitle("Distribution of Numerical Features")


for ax in plt.gcf().axes:
    ax.grid(False)

plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# 3. Stroke Cases Count
plt.figure(figsize=(6, 4))
sns.countplot(x="stroke", hue="stroke", data=df, palette="pastel", legend=False)  # Fix applied
plt.title("Count of Stroke Cases")
plt.xlabel("Stroke (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# 4. Boxplot of BMI by Stroke Status
plt.figure(figsize=(8, 6))
sns.boxplot(x="stroke", y="bmi", hue="stroke", data=df, palette="muted", dodge=False)
plt.title("BMI Distribution by Stroke Status")
plt.show()

# 5. Pairplot for Important Features Colored by Stroke
sns.pairplot(df, hue="stroke", vars=['age', 'avg_glucose_level', 'bmi'], palette="husl")
plt.show()

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np

# Define features and target
X = df.drop(columns=['stroke'])  # Features
y = df['stroke'].astype(int)  # Target variable

# Identify categorical and numerical columns
# Replace these with your actual column names
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ])

# Create KNN pipeline with preprocessing
knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define hyperparameters to tune
param_grid = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(knn_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Print best parameters
print("\nBest parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Evaluate the model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]  # For ROC AUC calculation

# Print classification metrics
print("\nTest set metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification report:\n", classification_report(y_test, y_pred))

print("\nConfusion matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Stroke', 'Stroke'],
            yticklabels=['No Stroke', 'Stroke'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#  Load & Clean the Data
df_fixed = pd.read_csv(file_url)  # Update path if needed

# Drop 'id' if it's present
if 'id' in df_fixed.columns:
    df_fixed = df_fixed.drop(columns=['id'])

# Define feature columns and target
X = df_fixed.drop(columns=['stroke'])  # Features
y = df_fixed['stroke'].astype(int)  # Target variable

# Identify Numerical & Categorical Features
numerical_features = ['age', 'avg_glucose_level', 'bmi']
categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married',
                        'work_type', 'Residence_type', 'smoking_status']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handling missing numerical values
    ('scaler', StandardScaler())  # Scaling numerical features
])



categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create Model Pipeline
knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

# Tune Hyperparameters
param_grid = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(knn_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Save the best model
best_model = grid_search.best_estimator_

#  Prediction on New Patient Data
feature_names = X_train.columns.tolist()

new_patient = pd.DataFrame([[
    0,    # gender (1=Male, 0=Female)
    65,   # age
    1,    # hypertension
    1,    # heart_disease
    1,    # ever_married (1=Yes, 0=No)
    2,    # work_type (Encoded)
    1,    # Residence_type (0=Rural, 1=Urban)
    100,  # avg_glucose_level
    38.5, # bmi
    3     # smoking_status (Encoded)
]], columns=feature_names)  # Ensure correct feature names

# Transform the new patient data
new_patient_transformed = best_model.named_steps['preprocessor'].transform(new_patient)

# Predict stroke risk
stroke_prediction = best_model.named_steps['classifier'].predict(new_patient_transformed)[0]
stroke_probability = best_model.named_steps['classifier'].predict_proba(new_patient_transformed)[:, 1][0]

print(f"Probability of Stroke: {stroke_probability:.4f} ({stroke_probability * 100:.2f}%)")


