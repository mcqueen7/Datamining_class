#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from category_encoders import TargetEncoder
from xgboost import XGBClassifier
# from catboost import CatBoostClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import warnings
import plotly.express as px
import plotly.graph_objects as go


from matplotlib.colors import LinearSegmentedColormap
import squarify


# In[2]:


# Set random seed for reproducibility
RS = 42
np.random.seed(RS)
warnings.filterwarnings("ignore")


# In[3]:


# Set color palette for visualizations
colors= ['#1c76b6', '#a7dae9', '#eb6a20', '#f59d3d', '#677fa0', '#d6e4ed', '#f7e9e5']
bi_palette = [colors[3], colors[0]]
sns.set_palette(colors)


# # Data Exploration & Understanding

# In[4]:


# Load datasets
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
print(f"Shape of train data: {df_train.shape}")
print(f"Shape of test data: {df_test.shape}")
# Initial Data Inspection
print("First 5 rows of train data:")
df_train.head()


# In[5]:


# Check for Duplicates
print(f"\nThere are {df_train.duplicated().sum()} duplicates in the training data.")
# Check for Missing Values
print("\nMissing values in each column of train data:")
print(df_train.isnull().sum())


# In[6]:


# Save test IDs and Drop unnecessary columns
test_ids = df_test['id']
df_train = df_train.drop(['id'], axis=1)
df_test = df_test.drop(['id'], axis=1)


# In[7]:


# Identify Target variable, categorical and numerical columns
TARGET_COL = 'Depression'
CATEGORICAL_COLS = df_train.select_dtypes(include=['object']).columns.tolist()
NUMERICAL_COLS = df_train.select_dtypes(exclude=['object']).columns.drop(TARGET_COL).tolist()


# In[8]:


print(f"\nTarget Column: {TARGET_COL}")
print(f"Categorical Columns: {CATEGORICAL_COLS}")
print(f"Numerical Columns: {NUMERICAL_COLS}")


# In[9]:


for column in CATEGORICAL_COLS:
    num_unique = df_train[column].nunique()
    print(f"'{column}' has {num_unique} unique categories.")


# In[10]:


# Print top 10 unique value counts for each categorical column
for column in CATEGORICAL_COLS:
    print(f"\nTop value counts in '{column}':\n{df_train[column].value_counts().head(10)}")


# In[11]:


# Skewness Analysis
print("\nSkewness of numerical columns:")
print(df_train[NUMERICAL_COLS].skew())


# # Exploratory Data Analysis

# In[12]:


import plotly.express as px
import pandas as pd

# 測試數據
df = pd.DataFrame({
    "x": [1, 2, 3, 4],
    "y": [10, 15, 13, 17]
})

fig = px.line(df, x="x", y="y", title="Test Plotly Graph")
fig.show()


# In[ ]:





# In[13]:


# Numerical Features Distribution
numerical_cols_to_plot = ["Age", "CGPA", "Work/Study Hours"]
def display_numerical_col_histogram(col):
    fig = px.histogram(df_train, x=col, title=f'Distribution of {col}', color=TARGET_COL)
    fig.show()
display_numerical_col_histogram("Age")


# In[14]:


display_numerical_col_histogram("CGPA")


# In[15]:


display_numerical_col_histogram("Work/Study Hours")


# In[ ]:





# In[16]:


# Categorical Features Distribution
categorical_cols_to_plot = ['Gender', 'Working Professional or Student', 'Academic Pressure',
                       'Work Pressure', 'Study Satisfaction', 'Job Satisfaction',
                       'Have you ever had suicidal thoughts ?', 'Financial Stress',
                       'Family History of Mental Illness']
def display_categorical_col_histogram(col):
    fig = px.histogram(df_train, x=col, title=f'Countplot of {col}', color=TARGET_COL)
    fig.show()
display_categorical_col_histogram('Gender')
display_categorical_col_histogram('Working Professional or Student')
display_categorical_col_histogram('Academic Pressure')
display_categorical_col_histogram('Work Pressure')
display_categorical_col_histogram('Study Satisfaction')
display_categorical_col_histogram('Job Satisfaction')
display_categorical_col_histogram('Have you ever had suicidal thoughts ?')
display_categorical_col_histogram('Financial Stress')
display_categorical_col_histogram('Family History of Mental Illness')


# In[17]:


# Explore Professions
value_counts = df_train['Profession'].value_counts()
sizes = value_counts.values[:20]
labels = [
    "Customer\nSupport" if label == "Customer Support" else
    "Marketing\nManager" if label == "Marketing Manager" else
    label
    for label in value_counts.index[:20]
]
fig = go.Figure(go.Treemap(
    labels = labels,
    parents = [''] * len(labels),
    values = sizes,
    marker_colors=colors
))
fig.update_layout(title=f"Treemap of Professions (Top 20)")
fig.show()


# In[18]:


top_n_professions = 20
profession_counts = df_train['Profession'].value_counts().nlargest(top_n_professions)
filtered_data = df_train[df_train['Profession'].isin(profession_counts.index)]
sankey_data = filtered_data.groupby(['Profession', 'Depression']).size().reset_index(name='Count')
labels = list(sankey_data['Profession'].unique()) + ['No Depression', 'Depression']
source_indices = []
target_indices = []
for _, row in sankey_data.iterrows():
    profession_index = labels.index(row['Profession'])
    depression_index = labels.index('Depression' if row['Depression'] == 1 else 'No Depression')
    source_indices.append(profession_index)
    target_indices.append(depression_index)
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color='blue'
    ),
    link=dict(
        source=source_indices,
        target=target_indices,
        value=sankey_data['Count']
    )
)])
fig.update_layout(
    title_text="Sankey Diagram of Profession and Depression",
    font_size=10,
    width=700,
    height=600)
fig.show()


# In[19]:


# Explore Degrees
value_counts = df_train['Degree'].value_counts()
sizes = value_counts.values[:20]
fig = go.Figure(go.Treemap(
    labels = value_counts.index[:20],
    parents = [''] * 20,
    values = sizes,
    marker_colors=colors
))
fig.update_layout(title=f"Treemap of Degrees (Top 20)")
fig.show()


# In[20]:


top_professions = df_train['Degree'].value_counts().nlargest(10).index
filtered_df = df_train[df_train['Degree'].isin(top_professions)]
agg_data = filtered_df.groupby(['Degree', 'Depression']).size().reset_index(name='Count')
fig = px.sunburst(agg_data,
                  path=['Degree', 'Depression'],
                  values='Count',
                  title='Sunburst Chart of Top 10 Degrees and Depression',
                  color='Count',
                  color_continuous_scale=px.colors.sequential.Oranges[:])
fig.show()


# In[21]:


# Target Variable Distribution
class_counts = df_train[TARGET_COL].value_counts().sort_index()
labels = ["No Depression", "Depression"]

fig = px.pie(names=labels, values=class_counts,
             title='Distribution of Target Variable', color_discrete_sequence=bi_palette)
fig.show()



# In[22]:


# Correlation between Variables
correlation_matrix = df_train.corr(numeric_only=True)
fig = px.imshow(correlation_matrix,
                text_auto=True,
                color_continuous_scale='RdYlBu',
                title='Heatmap of Correlation Matrix',
                aspect='auto')
fig.show()


# In[23]:


# Depression by Age and Work Pressure
df_train_copy = df_train.dropna(subset=['Age', 'Work Pressure', 'Depression'])
df_train_copy['Age_bin'] = pd.cut(df_train_copy['Age'], bins=10).astype(str)
df_train_copy['WorkPressure_bin'] = pd.cut(df_train_copy['Work Pressure'], bins=10).astype(str)
heatmap_data = df_train_copy.pivot_table(index='Age_bin', columns='WorkPressure_bin', values='Depression', aggfunc='mean')
fig = px.imshow(heatmap_data.values,
                labels=dict(x="Work Pressure Bin", y="Age Bin", color="Depression"),
                text_auto=True,
                color_continuous_scale='RdYlBu',
                title='Heatmap of Depression by Age and Work Pressure',
                aspect='auto')
fig.update_layout(
    yaxis=dict(
        tickmode='array',
        tickvals=list(range(len(heatmap_data.index))),
        ticktext=heatmap_data.index.astype(str).tolist()
    ),
    xaxis=dict(
        tickmode='array',
        tickvals=list(range(len(heatmap_data.columns))),
        ticktext=heatmap_data.columns.astype(str).tolist()
    )
)
fig.show()



# # Feature Engineering & Data Preprocessing

# In[24]:


# Feature Engineering
df_train['Age_WorkPressure'] = df_train['Age'] * df_train['Work Pressure']
df_test['Age_WorkPressure'] = df_test['Age'] * df_test['Work Pressure']


# In[25]:


encoder = TargetEncoder(cols=['City', 'Profession'])
df_train[['City_encoded', 'Profession_encoded']] = encoder.fit_transform(df_train[['City', 'Profession']], df_train[TARGET_COL])
df_test[['City_encoded', 'Profession_encoded']] = encoder.transform(df_test[['City', 'Profession']])


# In[26]:


# Define features and target
X_train = df_train.drop(TARGET_COL, axis=1)
y_train = df_train[TARGET_COL]


# In[27]:


# Redefine columns for preprocessing after feature engineering
NUMERICAL_COLS = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
CATEGORICAL_COLS = X_train.select_dtypes(include=['object']).columns.tolist()


# In[28]:


# Preprocessing Pipelines
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('convert_to_float32', FunctionTransformer(lambda x: x.astype(np.float32)))
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('ordinal', OrdinalEncoder(dtype=np.int32, handle_unknown='use_encoded_value', unknown_value=-1))
])


# In[29]:


# Combine the numerical and categorical pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, NUMERICAL_COLS),
        ('cat', categorical_pipeline, CATEGORICAL_COLS)
    ]
)


# In[30]:


# Apply Transformations
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(df_test)


# In[31]:


# Outlier Detection
isolation_forest = IsolationForest(contamination=0.04, random_state=RS)
outlier_labels = isolation_forest.fit_predict(X_train_preprocessed)
non_outliers_mask = outlier_labels != -1
X_train_preprocessed = X_train_preprocessed[non_outliers_mask]
y_train = y_train[non_outliers_mask]


# In[32]:


len(outlier_labels), sum(outlier_labels == -1)


# # Model Training & Evaluation

# In[33]:


# Define a scorer for all models.
scorer = make_scorer(accuracy_score)
model_performance = {}
def cross_validate_model(model, X, y, cv=5, scoring=scorer, model_name=None):
    start_time = time.time()
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, verbose=10)
    end_time = time.time()
    
    mean_accuracy = cv_scores.mean()
    std_accuracy = cv_scores.std()
    training_time = end_time - start_time

    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {mean_accuracy:.4f}")
    print(f"Standard Deviation of CV Accuracy: {std_accuracy:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    
    if model_name:
       model_performance[model_name] = {'mean_accuracy': mean_accuracy,
                                         'std_accuracy': std_accuracy,
                                         'training_time': training_time}
    return mean_accuracy


# In[34]:


# nn Neural Network
nn_model = MLPClassifier(hidden_layer_sizes=(64, 32),
                        activation='relu', solver='adam',
                        random_state=RS, max_iter=25, verbose=True, early_stopping=True)
start_time = time.time()

# Split data for training curves
X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(X_train_preprocessed, y_train, test_size=0.2, random_state=RS, stratify=y_train)
history = nn_model.fit(X_train_nn, y_train_nn)
end_time = time.time()

training_time = end_time - start_time

print("Neural Network results:")
mean_acc_nn = cross_validate_model(nn_model, X_train_preprocessed, y_train, model_name="Neural Network")


# In[35]:


# Plot training curves
if hasattr(history, 'loss_curve_'):
   fig = px.line(x=range(len(history.loss_curve_)), y=history.loss_curve_,
             title="Neural Network Training Loss Curve", labels={'x': 'Epoch', 'y': 'Training Loss'})
   fig.show()


# In[36]:


if hasattr(history, 'validation_scores_'):
    val_scores = history.validation_scores_
    fig = px.line(x=range(len(val_scores)), y=val_scores, title='Validation Accuracy Curve',
                  labels={'x': 'Epoch', 'y': 'Validation Accuracy'})
    fig.show()


# In[37]:


# Decision Tree
rf_model = RandomForestClassifier(n_estimators=100, random_state=RS, verbose=True)

print("Random Forest results:")
mean_acc_rf = cross_validate_model(rf_model, X_train_preprocessed, y_train, model_name="Random Forest")


# In[38]:


# Support Vector Machine (SVM)
svm_model = SVC(random_state=RS, max_iter=100, probability=True, verbose=True) # probability=True is necessary for ensemble

print("SVM results:")
mean_acc_svm = cross_validate_model(svm_model, X_train_preprocessed, y_train, model_name="SVM")


# In[39]:


# Gradient Boosting Machine
gbm_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=RS, verbose=True)

print("GBM results:")
cross_validate_model(gbm_model, X_train_preprocessed, y_train, model_name="GBM")


# In[40]:


# XGBoost
xgb_params = {
    'learning_rate': 0.298913248058474,
    'max_depth': 9,
    'min_child_weight': 3,
    'n_estimators': 673,
    'subsample': 0.5933970249700855,
    'gamma': 2.597137534750985,
    'reg_lambda': 0.11328048420927406,
    'colsample_bytree': 0.1381203919800721
}
xgb_model = XGBClassifier(**xgb_params, use_label_encoder=False, random_state=RS, verbose=True)

print("XGBoost results:")
cross_validate_model(xgb_model, X_train_preprocessed, y_train, model_name="XGBoost")


# In[46]:


# Ensemble Methods
 # Stacking Ensemble 
stacking_ensemble = StackingClassifier(
    estimators=[
        ('nn', nn_model),
        ('rf', rf_model),
        ('gbm', gbm_model),
        ('xgb', xgb_model),
    ],
    final_estimator=LogisticRegression(),
    passthrough=False, 
    verbose=True
)

print("Stacking Ensemble Results:")
cross_validate_model(stacking_ensemble, X_train_preprocessed, y_train, model_name="Stacking Ensemble")


# In[42]:


# Model Comparison Plot
model_names = list(model_performance.keys())
mean_accuracies = [model_performance[name]['mean_accuracy'] for name in model_names]
training_times = [model_performance[name]['training_time'] for name in model_names]



fig = go.Figure()
fig.add_trace(go.Bar(x=model_names, y=mean_accuracies, name='Mean Accuracy', marker_color=colors[0]))
fig.add_trace(go.Bar(x=model_names, y=training_times, name='Training Time (s)', marker_color=colors[2],
                   yaxis='y2'))
fig.update_layout(title="Model Performance Comparison",
                  xaxis_title="Model",
                  yaxis_title="Mean Accuracy",
                    yaxis2=dict(
                        title='Training Time (s)',
                        overlaying='y',
                        side='right'
                    ))

fig.show()


# # Final Model Training & Prediction

# In[48]:


# stacking_ensemble.fit(X_train_preprocessed, y_train)
test_preds = stacking_ensemble.predict(X_test_preprocessed)

output = pd.DataFrame({'id': test_ids, 'class': test_preds})
output.to_csv('submission.csv', index=False)
output.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




