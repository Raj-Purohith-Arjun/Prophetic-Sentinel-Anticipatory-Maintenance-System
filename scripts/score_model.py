import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score, precision_score, recall_score
import logging

def get_file(f):

    f = Path(f)
    if f.is_file():
        return f
    else:
        files = list(f.iterdir())
        if len(files) == 1:
            return files[0]
        else:
            raise Exception("********This path contains more than one file*******")

def parse_args():
    # setup argparse
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--input_data", type=str, help="path containing data for scoring"
    )
    parser.add_argument(
        "--input_model", type=str, help="input path for model"
    )

    parser.add_argument(
        "--output_data", type=str,  help="output predictions for model"
    )

    # parse args
    args = parser.parse_args()

    # return args
    return args




def score(input_data, input_model):

    test_file = get_file(input_data)
    data_test = pd.read_csv(test_file)

    # Load model
    model = mlflow.pyfunc.load_model(input_model)



    # Score model using test data
    y_pred = model.predict(data_test)
    
  
    print("Model Scored Successfully")

    mlflow.sklearn.log_model(model, "rf_model")

    pred_df = pd.DataFrame(y_pred, columns=['Failure'])

 
    print('Preprocessing Done')

    
    return data_test.assign(Failure=pred_df['Failure'])

def main(args):
    score(args.input_data, args.input_model)
 

    



# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # call main function
    main(args)


