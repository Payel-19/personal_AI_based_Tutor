import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import PyPDF2
import os
import webbrowser
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
file_path = "large dataset for predictions.csv"
df = pd.read_csv(file_path)

# Exploratory Data Analysis (EDA)
print("Basic Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Visualize feature distributions
plt.figure(figsize=(12,6))
sns.histplot(df['Assessment Score'], bins=30, kde=True)
plt.title("Distribution of Assessment Scores")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))

numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Gender', 'Country', 'State', 'City', 'Parent Occupation', 'Earning Class',
                       'Level of Student', 'Level of Course', 'Course Name']
for col in categorical_columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Define features and target
X = df[['Age', 'Gender', 'Country', 'State', 'City', 'Parent Occupation', 'Earning Class',
        'Level of Student', 'Level of Course', 'Course Name', 'Study Time Per Day', 'IQ of Student']]
y = df['Assessment Score']

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

df['Predicted Score'] = model.predict(X)

# Determine promotion status
df['Promotion Status'] = df['Predicted Score'].apply(lambda x: 'Promoted' if x >= 50 else 'Not Promoted')

# Ensure 'Level of Student' is numeric
df['Level of Student'] = pd.to_numeric(df['Level of Student'], errors='coerce')

def filter_material(level):
    if pd.isna(level):
        return "Unknown"
    elif level < 4:
        return "Basic Materials"
    elif 4 <= level <= 7:
        return "Intermediate Materials"
    else:
        return "Advanced Materials"

df['Recommended Material'] = df['Level of Student'].apply(lambda x: filter_material(x))

# Save results
df.to_excel("student_predictions.xlsx", index=False)
print("Updated results saved to student_predictions.xlsx")

# Display results
print(df[['Name', 'Assessment Score', 'Predicted Score', 'Promotion Status', 'Recommended Material']].head())

# Function to provide PDF and video explanations
def provide_material():
    default_pdf = "M1_Data Warehousing.pdf"
    explained_pdf = "M2_Data Warehousing.pdf"
    default_video = "https://youtu.be/gmvvaobm7eQ"
    explained_video = "https://youtu.be/J_LnPL3Qg70"

    # Ensure the default PDF exists and open it
    if os.path.exists(default_pdf):
        print("Opening default PDF...")
        webbrowser.open(default_pdf)  # Open the default PDF
    else:
        print(f"Default PDF '{default_pdf}' not found in the current folder.")

    # Open the default video link
    print(f"Click to watch default video: {default_video}")
    webbrowser.open(default_video)  # Open the video in the default browser

    while True:
        user_input = input("Do you need a detailed explanation? (Yes/No): ").strip().lower()
        if user_input == "yes":
            # Open detailed PDF if it exists
            if os.path.exists(explained_pdf):
                print("Opening detailed PDF...")
                webbrowser.open(explained_pdf)  # Open the detailed PDF
            else:
                print(f"Detailed PDF '{explained_pdf}' not found in the current folder.")

            # Open the detailed video link
            print(f"Click to watch detailed video: {explained_video}")
            webbrowser.open(explained_video)  # Open the video in the default browser
            break
        elif user_input == "no":
            print("Okay, no detailed explanation provided.")
            break
        else:
            print("Invalid input. Please enter 'Yes' or 'No'.")

# Call the function to provide PDFs and videos
provide_material()
