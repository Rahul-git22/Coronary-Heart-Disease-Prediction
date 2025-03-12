# Coronary-Heart-Disease-Prediction

This project aims to predict the risk of Coronary Heart Disease (CHD) based on various health indicators using machine learning. The dataset used is the Framingham Heart Study dataset, which contains various medical attributes related to heart disease.

## Project Overview
The goal of this project is to develop a machine learning model that can predict the risk of Coronary Heart Disease (CHD) based on a set of health indicators. The project involves the following steps:
1. Data Loading and Initial Exploration
2. Data Preprocessing (Handling Missing Values, Feature Engineering)
3. Exploratory Data Analysis (EDA)
4. Model Training and Evaluation
5. Saving the Model and Preprocessing Pipeline for Future Use
6. Creating a Streamlit App for Real-Time Predictions

## Dataset
The dataset used in this project is the Framingham Heart Study dataset. It contains the following columns:
- **male**: Gender (1 = male, 0 = female)
- **age**: Age in years
- **education**: Education level (1 = High School, 2 = College, 3 = University, 4 = Advanced Degree)
- **currentSmoker**: Whether the patient is a current smoker (1 = yes, 0 = no)
- **cigsPerDay**: Number of cigarettes smoked per day
- **BPMeds**: Whether the patient is on blood pressure medication (1 = yes, 0 = no)
- **prevalentStroke**: Whether the patient has had a stroke (1 = yes, 0 = no)
- **prevalentHyp**: Whether the patient has hypertension (1 = yes, 0 = no)
- **diabetes**: Whether the patient has diabetes (1 = yes, 0 = no)
- **totChol**: Total cholesterol level
- **sysBP**: Systolic blood pressure
- **diaBP**: Diastolic blood pressure
- **BMI**: Body Mass Index
- **heartRate**: Heart rate
- **glucose**: Glucose level
- **TenYearCHD**: Target variable indicating the risk of CHD within 10 years (1 = risk, 0 = no risk)

## Dependencies
To run this project, you need the following dependencies:
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost
- streamlit
- pickle

You can install the required dependencies using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost streamlit
```

## Setup and Installation
1. Clone the repository or download the project files.
2. Ensure you have the required dependencies installed.
3. Place the framingham.csv dataset in the same directory as the project script.
4. Ensure the app.py file is in the same directory as the project script.

## Running the Project
To run the project, execute the following command in your terminal:

```
streamlit run app.py
```
This command will:
1. Start the Streamlit server.
2. Open the Streamlit app in your default web browser.
3. The app will load the trained model and preprocessing pipeline from chd_pipeline.pkl.
4. You can input the required health indicators, and the app will predict the risk of CHD.

## Model Pipeline
The pipeline consists of the following steps:
1. **Numerical Pipeline:**
- KNN Imputer to handle missing values.
- Log1p transformation to handle skewness.
- Winsorization to handle outliers.
- Standard Scaling to normalize the data.

2. **Categorical Pipeline**:
- KNN Imputer to handle missing values.

3. **Model**:
- Random Forest Classifier for predicting the risk of CHD.

## Results
The script will output the training and test accuracy of the model. The trained pipeline will be saved to  ```chd_pipeline.pkl```.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

