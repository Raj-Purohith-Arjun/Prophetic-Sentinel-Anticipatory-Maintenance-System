import argparse
from pathlib import Path
import os
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
import sys
import timeit
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure callers send all required parameters
# and that they are of the right type
parser = argparse.ArgumentParser("prep")
parser.add_argument("--input_data", type=str, help="Name of the folder containing input data for the operation")
parser.add_argument("--output_data_x_train", type=str, help="Name of folder we will write training results out to")
parser.add_argument("--output_data_x_test", type=str, help="Name of folder we will write test results out to")
parser.add_argument("--output_data_y_train", type=str, help="Name of folder we will write training results out to")
parser.add_argument("--output_data_y_test", type=str, help="Name of folder we will write test results out to")

args=parser.parse_args()

# Diagnostic print statements 
print('Preparing data...')

# State the input and output data directories for print
lines = [

f"input_data: {args.input_data}",
f"output_data_x_train: {args.output_data_x_train}"
f"output_data_x_test: {args.output_data_x_train}"
f"output_data_y_train: {args.output_data_y_train}"
f"output_data_y_test: {args.output_data_y_test}"
]
for line in lines:
    print(line)

# View the contents of the input data directory
print(os.listdir(args.input_data))  

file_list=[]
for filename in os.listdir(args.input_data):
    print("Loading file: %s..." %filename)
    with open(os.path.join(args.input_data,filename),"r") as f:
        input_df = pd.read_csv((Path(args.input_data) / filename))
        file_list.append(input_df)

# Concatenate the dataframes
df=pd.concat(file_list,ignore_index=True)


def prepare_data(data):

    # Encode categorical data
    categorical_features = 'Type'

    # Label encoding
    LE = LabelEncoder()
    data[categorical_features] = LE.fit_transform(data[categorical_features])


    # Extract features and target
    X = data.drop('Machine failure', axis=1)
    y = data['Machine failure']


    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

    # Convert to DataFrames 
    X_train_df = pd.DataFrame(X_train, columns=X.columns)
    X_test_df = pd.DataFrame(X_test, columns=X.columns)

    Numerical_features = ['Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    ]
    
    # Standard Scaler for feature scaling numerical data only
    Scaler = StandardScaler((-1,1))
    X_train_df[Numerical_features] = Scaler.fit_transform(X_train_df[Numerical_features])
    X_test_df[Numerical_features] = Scaler.transform(X_test_df[Numerical_features])
    
    y_train_df = pd.DataFrame(y_train, columns=['Machine failure'])
    y_test_df = pd.DataFrame(y_test, columns=['Machine failure'])

    return X_train_df , X_test_df, y_train_df, y_test_df

# Preprocess the data
X_train , X_test, y_train, y_test = prepare_data(df)

# Save the preprocessed data to the output data directory
output_path_x_train = Path(args.output_data_x_train) / "x_train.csv"
output_path_x_test = Path(args.output_data_x_test) / "x_test.csv"
output_path_y_train = Path(args.output_data_y_train) / "y_train.csv"
output_path_y_test = Path(args.output_data_y_test) / "y_test.csv"

# Save the preprocessed data to the output data directory
X_train.to_csv(output_path_x_train, index=False)
X_test.to_csv(output_path_x_test, index=False)
y_train.to_csv(output_path_y_train, index=False)
y_test.to_csv(output_path_y_test, index=False)

print('Preparation Done')

