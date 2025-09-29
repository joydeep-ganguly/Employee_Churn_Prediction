import streamlit as st
import requests
import matplotlib.pyplot as plt
import pickle
import pandas as pd

BASE_URL = 'http://127.0.0.1:8000'

with open("category_mapping.pkl", 'rb') as f:
        category_mapping = pickle.load(f)

st.title('Employee Attrition Prediction')

input_method = st.radio('Choose input method:', ('Upload CSV', 'Manual Input'))

if input_method=='Upload CSV':
  uploaded_file = st.file_uploader('Upload CSV file',type='csv')
  
  if uploaded_file:
    filename = str(uploaded_file.name)
    #for request body pass as json
    #response = requests.post(f'{BASE_URL}/predictfile',json = {'fname' : filename})
    #for path/query parameters pass as url format
    response = requests.post(f'{BASE_URL}/predictfile?fname={filename}')

    st.subheader("Analysis Results")
    if response.status_code==200:
      response_dict = response.json()
      response_df = pd.DataFrame.from_dict(response_dict['file'])

      # Metrics
      col1, col2, col3 = st.columns(3)
      total = len(response_df)
      stay = len(response_df[response_df['Prediction']=='Stay'])
      leave = len(response_df[response_df['Prediction']=='Leave'])
      col1.metric("Total Employees", total)
      col2.metric("Employees Staying", stay, f"{stay/total:.1%}")
      col3.metric("Employees Leaving", leave, f"{leave/total:.1%}")

      # Pie chart
      fig, ax = plt.subplots()
      ax.pie([stay, leave], 
            labels=['Stay', 'Leave'], 
            colors=['#4CAF50', '#F44336'],
            autopct='%1.1f%%')
      ax.set_title("Attrition Distribution")
      st.pyplot(fig)

      #Show Data
      st.subheader('Detailed Prediction')
      st.write(response_df)

    else:
      error = response.json()
      st.write(error['detail'])     
else:
  st.subheader('Employeee Details')
  no_of_projects = st.number_input('Number of Projects', min_value=0, max_value=20, value=2)
  salary = st.number_input('Salary', min_value=40000, max_value=130000, value=50000)
  age = st.slider('Age',21,61,30)
  tenure = st.slider('Tenure (Years)',0,29,2)
  #get the category lists from mapping
  dept_name = st.selectbox('Department Name',category_mapping['dept_name'])
  title =  st.selectbox('Job Title',category_mapping['title'])
  sex = st.selectbox('Gender',category_mapping['sex'])
  rating = st.selectbox('Performance Rating',category_mapping['Last_performance_rating'])

  if st.button('Predict'):
    response = requests.post(
      f'{BASE_URL}/predict',
        json={
              'age' : age,
              'tenure' : tenure,
              'salary' : salary,
              'no_of_projects' : no_of_projects,
              'dept_name' : dept_name,
              'title' : title,
              'sex' : sex,
              'Last_performance_rating' : rating
          }
    )
    st.subheader('Results')

    if response.status_code==200:
      prediction = response.json()
      if prediction['class']=='1':
        st.write('The employee will leave.')
      else:
        st.write('The employee will stay.')
    else:
      st.write('Failed to predict')
      