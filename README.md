# Employee Churn Prediction

![alt text]('./empchurn.jpg')
This repository contains an implementation of an **Employee Churn Prediction** system using **FastAPI** as the backend and **Streamlit** as the frontend. The system predicts whether an employee is likely to stay or leave based on various features such as age, tenure, salary, and more.

---

## Features

- **FastAPI Backend**:

  - Provides endpoints for single employee prediction and batch prediction via CSV files.
  - Preprocesses input data to align with the trained model.
  - Utilizes a pre-trained Random Forest model for predictions.

- **Streamlit Frontend**:
  - User-friendly interface for manual input or CSV file upload.
  - Displays prediction results, including metrics, pie charts, and detailed data tables.

---

## Project Structure

```
.
├── app.py                  # Streamlit frontend
├── deployapi.py            # FastAPI backend
├── random_forest_model.pkl # Pre-trained Random Forest model
├── scaler.pkl              # Scaler for feature normalization
├── model_columns.pkl       # Model columns for input alignment
├── category_mapping.pkl    # Mapping for categorical features
├── testdata.csv            # Sample test data
```

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/employee-churn-prediction.git
   cd employee-churn-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. ## Ensure the following files are present:

random_forest_model.pkl

- scaler.pkl

- model_columns.pkl

- category_mapping.pkl

---

## Usage

### 1. Start the FastAPI Backend

Run the FastAPI backend to serve prediction endpoints:

```bash
uvicorn deployapi:app --reload
```

The backend will be available at `http://127.0.0.1:8000`.

### 2. Start the Streamlit Frontend

Run the Streamlit app for the user interface:

```bash
streamlit run app.py
```

The frontend will be available at `http://localhost:8501`.

---

## Endpoints (FastAPI)

- **`POST /predict`**: Predicts churn for a single employee.

  - Request Body:
    ```json
    {
      "age": 30,
      "tenure": 5,
      "salary": 60000,
      "no_of_projects": 3,
      "dept_name": "Sales",
      "title": "Staff",
      "sex": "M",
      "Last_performance_rating": "A"
    }
    ```
  - Response:
    ```json
    {
      "class": "1"
    }
    ```

- **`POST /predictfile`**: Predicts churn for multiple employees from a CSV file.
  - Query Parameter:

fname

(path to the CSV file)

- Response:
  ```json
  {
    "file": [
      {
        "age": 30,
        "Prediction": "Stay",
        "Probability": 0.2
      },
      ...
    ]
  }
  ```

---

## Frontend Features (Streamlit)

- **Manual Input**:

  - Enter employee details such as age, salary, and department.
  - Get instant predictions on whether the employee will stay or leave.

- **CSV Upload**:
  - Upload a CSV file with employee data.
  - View detailed predictions, metrics, and visualizations.

---

## Sample Data

A sample CSV file (`testdata.csv`) is included for testing batch predictions. The file contains columns such as

age,tenure,salary,dept_name and more.

---

## Preprocessing

The backend preprocesses input data to match the model's training data:

- Bins

age,tenure, and salary into categorical groups.

- Encodes categorical features using one-hot encoding.
- Scales numerical features using a pre-trained scaler.

---

## Dependencies

- **Backend**:

  - FastAPI
  - scikit-learn
  - pandas
  - numpy
  - joblib

- **Frontend**:
  - Streamlit
  - matplotlib

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

- **FastAPI** for the backend framework.
- **Streamlit** for the interactive frontend.
- **scikit-learn** for the machine learning model.

---

Feel free to contribute to this project by submitting issues or pull requests!
