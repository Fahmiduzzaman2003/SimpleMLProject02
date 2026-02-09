import gradio as gr 
import pickle
import pandas as pd 
import numpy as np 


with open("best_model.pkl","rb")as file:
    model=pickle.load(file)


def predict(gender,age,salary):
    salary_per_age=salary/age
    age_group="Young"if age<=30 else ("Mid" if age<=45 else "Old")
    data=pd.DataFrame(
        [
            {
                "Age":age,
                "EstimatedSalary":salary,
                "SalaryPerAge":salary_per_age,
                "Gender":gender,
                "AgeGroup":age_group


            }
        ]
    )
    prediction=model.predict(data)[0]
    return "Will Purchase" if prediction==1 else "Will Not Purchase"

grd=gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(["Male","Female"],label="Gender"),
        gr.Slider(10,100,value=90,label="Age"),
        gr.Slider(10000,200000,value=50000,label="Estimated Salary")
    ],
    outputs=gr.Text(label="Prediction"),
    title="Social Network Ad Purchase Predictor",
    description="Predict whether a user will purchase a product based on their demographics."
)

grd.launch()

