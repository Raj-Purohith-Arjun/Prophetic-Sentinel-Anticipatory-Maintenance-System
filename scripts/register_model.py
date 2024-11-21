import argparse
import pandas as pd
import os
import json
from pathlib import Path
import mlflow
import azureml
from azureml.core import Workspace, Experiment, Run
from azureml.core.model import Model


print("Register model...")
mlflow.start_run()

parser=argparse.ArgumentParser("register")
parser.add_argument("--model", type=str, help="Path to trained model")
parser.add_argument("--evaluation_report", type=str, help="Path of model's Evaluation report")

args=parser.parse_args()

lines=[
    f"Model path: {args.model}",
    f"Test report path: {args.evaluation_report}",
]

for line in lines:
    print(line)

# 

run = Run.get_context()

ws=run.experiment.workspace

print('Loading Model...')
model = mlflow.sklearn.load_model(Path(args.model))

print('Loading Evaluation Report')
fname='results.json'
with open(Path(args.evaluation_report) / fname,'r') as fp:
    results=json.load(fp)

print('Saving Model Locally')
root_model_path = 'Trained_Models'
os.makedirs(root_model_path, exist_ok=True)

mlflow.sklearn.save_model(model, root_model_path)

# Azure ML Model Registry

registered_model_name = 'FailurePredictionModel'
model_description = 'Random Forest Classifier for Failure Prediction of Manufacturing Machines'

print('Registering Model... test')
registered_model  = Model.register(model_path = root_model_path,
                                   model_name = registered_model_name,
                                   tags=results,
                                   description = model_description,
                                   workspace = ws)


mlflow.end_run()
print('Model Registery Done')
