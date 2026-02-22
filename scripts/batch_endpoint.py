import os
import glob
import mlflow
import pandas as pd
from azureml.core import Run
from azureml.core.model import Model

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def init():
    global model
    global device
    model_name = "FailurePredictionModel"
    run = Run.get_context()
    ws = run.experiment.workspace
    print('Loading model...')
    model_path = Model.get_model_path(model_name=model_name, _workspace=ws, version=3)
    model = mlflow.pyfunc.load_model(model_path)
    print('Getting Model Path')



def run(mini_batch):
    print(f"run method start: {__file__}, run({len(mini_batch)} files)")

    data = pd.concat(
        map(
            lambda fp: pd.read_csv(fp), mini_batch
        )
    )  
 
   
    # Prepare the data
    print('Preparing data...')
    data  = process_data(data)
    print('Data is ready for prediction')
    # Make predictions and return the results
    print('Making predictions...')
    pred = model.predict(data)
    pred = pd.DataFrame(pred, columns=['Machine Failure Prediction'])
    return data.assign(MachineFailed=pred['Machine Failure Prediction'])
   


