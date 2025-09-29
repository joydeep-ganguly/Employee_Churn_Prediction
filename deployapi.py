from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import StandardScaler

def load_resources():
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    model_columns = joblib.load("model_columns.pkl")
    
    return model, scaler, model_columns

def preprocess_input(input_df, model_columns, scaler):
   #Preprocess input data to match the training data
   #create age group bins
  age_bins = [21, 29, 37, 46, 54, 62]
  age_labels = ['21-28', '29-36', '37-45', '46-53', '54-62']
  input_df['age_group'] = pd.cut(input_df.age, bins=age_bins, labels=age_labels, right=False)
  input_df['age_group'] = input_df['age_group'].astype('object')
  #create tenure_group bin
  tenure_bins = [0, 6, 12, 18, 24, 30]
  tenure_labels = ['0-5', '6-11', '12-17', '18-23', '24-30']
  input_df['tenure_group'] = pd.cut(input_df.tenure, bins=tenure_bins, labels=tenure_labels, right=False)
  input_df['tenure_group'] = input_df['tenure_group'].astype('object')
  #create salary bin
  salary_bins = [40000,60000,80000,100000,130000]
  salary_labels=['low', 'medium', 'high', 'very_high']
  input_df['salary_bin'] = pd.cut(input_df['salary'], bins=salary_bins, labels=salary_labels,right=False)
  input_df['salary_bin']= input_df['salary_bin'].astype('object')
  #select features
  features = ['no_of_projects', 'salary_bin', 'age_group', 'tenure_group', 
            'dept_name', 'title', 'sex', 'Last_performance_rating']  
  processed_df = input_df[features]
  #Encoding of columns
  encoded_df = pd.get_dummies(processed_df, drop_first=True)
  #Align the encoded columns with the model columns
  missing_cols = set(model_columns) - set(encoded_df.columns)
  for col in missing_cols:
     encoded_df[col] = 0
  encoded_df = encoded_df[model_columns]

  scaled_df = scaler.transform(encoded_df)
  return scaled_df

#Class creation for defining input variables
class request_body(BaseModel):
   age : float
   tenure : float
   salary : float
   no_of_projects : int
   dept_name : object
   title : object
   sex : object
   Last_performance_rating: object

class request_filename(BaseModel):
   mfile : str

app = FastAPI()

@app.post('/predict')
def singledata(data : request_body):
  model, scaler, model_columns = load_resources()

  input_data = {
     'age' : data.age,
     'tenure' : data.tenure,
     'salary' : data.salary,
     'no_of_projects' : data.no_of_projects,
     'dept_name' : data.dept_name,
     'title' : data.title,
     'sex' : data.sex,
     'Last_performance_rating' : data.Last_performance_rating
  }
  input_df = pd.DataFrame([input_data])

  # Pass model_columns and scaler to preprocess_input
  processed_data = preprocess_input(input_df, model_columns, scaler)
  prediction = model.predict(processed_data)[0]
  return {'class' : str(prediction)}

# using query or path parameters, can also use request body (BaseModel) 
# def filedata(fname : request_filename):
# df = pd.read_csv(fname.mfile) 

@app.post('/predictfile')
def filedata(fname : str):
   model, scaler, model_columns = load_resources()

   df = pd.read_csv(fname)
   required_cols = ['age', 'tenure', 'salary', 'dept_name', 'title', 
                    'sex', 'Last_performance_rating', 'no_of_projects']
   #check if columns matching with the csv
   if all(col in df.columns for col in required_cols):
      df = df[required_cols]
      # Pass model_columns and scaler to preprocess_input
      processed_data = preprocess_input(df, model_columns, scaler)
      predictions = model.predict(processed_data)
      probabilities = model.predict_proba(processed_data)[:, 1]
      df['Prediction'] = predictions
      df['Prediction'] = ['Leave' if p==1 else 'Stay' for p in predictions]
      df['Probability'] = probabilities

      return {'file' : df.to_dict(orient='records')} 
      
   
@app.get('/')
def main():
  return {'message': 'Welcome to Fast API!'}

@app.get('/{name}')
def hello_name(name : str):
  return {'message': f'Welcome to Fast API! {name}'}

