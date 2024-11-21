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


# Define the preprocessing function
def preprocess_data(data):
    print('preprocess function')
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
    
    return data
# Ensure callers send all required parameters
# and that they are of the right type
parser = argparse.ArgumentParser("prep")
parser.add_argument("--input_data", type=str, help="Name of the folder containing input data for the operation")
parser.add_argument("--output_data", type=str, help="Name of the folder containing output data for the operation")

args=parser.parse_args()

# Diagnostic print statements 
print('Preprocessing data...')

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


# Concatenate the dataframes
df=pd.concat(file_list)

# Preprocess the data
df = preprocess_data(df)
print("Df Preprocessing Done")
# Save the preprocessed data to the output data directory

output_path = Path(args.output_data) / "preprocessed_data.csv"
df.to_csv(output_path, index=False)
print('Preprocessing Done')


