
# Diabetes Prediction AI API

## Index

- [Frontend](https://github.com/FabsSWD/diabetes-app-frontend)
- [Backend](https://github.com/FabsSWD/diabetes-service-backend)
- [AI API](https://github.com/FabsSWD/diabetes-ml)

## Project Description

This project is a **Machine Learning API** that predicts the likelihood of diabetes based on user-provided health data. The API is built using **Flask** and a pre-trained **Logistic Regression model** that processes various medical and personal metrics to make predictions. This API serves as the AI component in a larger diabetes application ecosystem, interfacing with a **React frontend** and a **user management backend**.

## Technologies Used

- **Python 3.8**: Main programming language.
- **Flask**: Framework for creating the REST API.
- **Pandas**: For data manipulation and processing.
- **Scikit-Learn**: For data preprocessing and model handling.
- **Keras**: For deep learning model integration (Logistic Regression model saved in `.keras` format).
- **Pickle**: Used for saving and loading the preprocessing objects (Scaler and Imputer).

## Project Structure

The project is organized with separate files for API functionality, ML model, and preprocessing objects:

- `app.py`: Main file containing the Flask API setup and endpoints for prediction.
- `IA Model.ipynb`: Jupyter Notebook where the machine learning model was trained, saved, and evaluated.
- `diabetes.csv`: Dataset used for model training and testing.
- `imputer.pkl`: A saved Scikit-Learn `SimpleImputer` object for handling missing values.
- `scaler.pkl`: A saved Scikit-Learn `StandardScaler` object for data normalization.
- `logistic_regression_model.keras`: Trained Logistic Regression model saved in Keras format.

## Features

The API provides the following functionalities:

### 1. Predict Diabetes

- **Endpoint**: `/predict`
- **HTTP Method**: `POST`
- **Description**: Receives user health metrics and returns a prediction indicating whether the user is likely to have diabetes.
- **Request Body**:
  ```json
  {
    "pregnancies": 2,
    "glucose": 120,
    "bloodPressure": 80,
    "skinThickness": 25,
    "insulin": 85,
    "bmi": 28.0,
    "diabetesPedigreeFunction": 0.5,
    "age": 45
  }
  ```
- **Response**:
  ```json
  {
    "diabetes_prediction": "positive"
  }
  ```
  or
  ```json
  {
    "diabetes_prediction": "negative"
  }
  ```

## Main Files

### 1. `app.py`
This is the main Flask application file, containing the `/predict` endpoint, which handles incoming prediction requests. It loads the pre-trained model, scaler, and imputer, applies preprocessing to the input data, and returns a prediction.

### 2. `IA Model.ipynb`
This Jupyter notebook is used for model training and evaluation. It covers the following steps:
   - Data loading and preprocessing.
   - Model training using Logistic Regression.
   - Model evaluation and saving of trained model and preprocessing objects.

### 3. `diabetes.csv`
This dataset contains the health records used for training and testing the model. It includes fields such as `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, and `Age`.

### 4. `imputer.pkl` and `scaler.pkl`
   - **imputer.pkl**: This file stores a `SimpleImputer` object, which handles any missing values in the input data.
   - **scaler.pkl**: This file stores a `StandardScaler` object, which normalizes the input data.

### 5. `logistic_regression_model.keras`
This file contains the saved Logistic Regression model, which processes preprocessed input data to predict the likelihood of diabetes.

## How to Run the Project

### Prerequisites

- **Python 3.8+**
- **Flask** and required dependencies

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/FabsSWD/diabetes-ml
   cd diabetes-ml
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the Flask API server:
   ```bash
   python app.py
   ```

4. The API will be accessible at `http://localhost:5000/predict`.

## Testing the API

You can test the API using **Postman** or **cURL**:

### Example with cURL:

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{
  "pregnancies": 2,
  "glucose": 120,
  "bloodPressure": 80,
  "skinThickness": 25,
  "insulin": 85,
  "bmi": 28.0,
  "diabetesPedigreeFunction": 0.5,
  "age": 45
}'
```

### Expected Response

```json
{
  "diabetes_prediction": "positive" // or "negative"
}
```

## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this project for both personal and commercial purposes.
