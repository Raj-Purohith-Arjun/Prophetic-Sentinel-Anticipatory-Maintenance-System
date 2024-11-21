
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

parser = argparse.ArgumentParser("prep")
parser.add_argument("--input_data", type=str, help="Name of the folder containing input data for the operation")
parser.add_argument("--output_data", type=str, help="Name of the folder containing output data for the operation")

args=parser.parse_args()


# State the input and output data directories for print
lines = [

f"input_data: {args.input_data}",
f"output_data: {args.output_data}",
]

for line in lines:
    print(line)

# View the contents of the input data directory
print(os.listdir(args.input_data))  

file_list=[]
for filename in os.listdir(args.input_data):
    print("Loading file: %s..." % filename)
    with open(os.path.join(args.input_data,filename),"r") as f:
        input_df = pd.read_csv((Path(args.input_data) / filename))
        file_list.append(input_df)
df = pd.concat(file_list)



def process_data(data):

    # Select columns to drop
    columns_to_drop = ['UDI', 'Product ID']

    # Drop columns with no predictive power
    data.drop(columns=columns_to_drop, axis=1, inplace=True)
    
    print('Data Null: \n', data.isnull().sum())

    if data.isnull().sum().sum() == 0:
        print('Data has no null values')
    
    else:
        print('Data has null values')
        data.dropna(axis=0,inplace=True)
    
    # Drop duplicates
    data = data.drop_duplicates()
   
    # Encode categorical data
    categorical_features = 'Type'

    # Label encoding
    LE = LabelEncoder()
    data[categorical_features] = LE.fit_transform(data[categorical_features])

    Numerical_features = ['Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    ]
    
    # Standard Scaler for feature scaling numerical data only
    Scaler = StandardScaler()
    data[Numerical_features] = Scaler.fit_transform(data[Numerical_features])
    return data


# Preprocess the data
df = process_data(df)

print("Df Preprocessing Done")
# Save the preprocessed data to the output data directory

output_path = Path(args.output_data) / "preprocessed_data.csv"
df.to_csv(output_path, index=False)
print('Preprocessing Done')
