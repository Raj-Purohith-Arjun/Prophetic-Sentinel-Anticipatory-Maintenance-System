
import mlflow
import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import os
import json



def main(args):


    print('listing:::')
    # x_train 
    print(os.listdir(args.x_train))

    x_train_file_list=[]
    for filename in os.listdir(args.x_train):
        print("Reading file: %s ..." % filename)
        with open(os.path.join(args.x_train, filename), "r") as f:
            input_df=pd.read_csv((Path(args.x_train) / filename))
            x_train_file_list.append(input_df)

    # Concatenate the list of Python DataFrames
    x_train_df =pd.concat(x_train_file_list)

    print("1 Done")



    # x_test
    print(os.listdir(args.x_test))

    x_test_file_list=[]
    for filename in os.listdir(args.x_test):
        print("Reading file: %s ..." % filename)
        with open(os.path.join(args.x_test, filename), "r") as f:
            input_df=pd.read_csv((Path(args.x_test) / filename))
            x_test_file_list.append(input_df)

    # Concatenate the list of Python DataFrames
    x_test_df =pd.concat(x_test_file_list)

  

    # y_train
    print(os.listdir(args.y_train))

    y_train_file_list=[]
    for filename in os.listdir(args.y_train):
        print("Reading file: %s ..." % filename)
        with open(os.path.join(args.y_train, filename), "r") as f:
            input_df=pd.read_csv((Path(args.y_train) / filename))
            y_train_file_list.append(input_df)

    # Concatenate the list of Python DataFrames
    y_train_df =pd.concat(y_train_file_list)



    # y_test
    print(os.listdir(args.y_test))

    y_test_file_list=[]
    for filename in os.listdir(args.y_test):
        print("Reading file: %s ..." % filename)
        with open(os.path.join(args.y_test, filename), "r") as f:
            input_df=pd.read_csv((Path(args.y_test) / filename))
            y_test_file_list.append(input_df)

    # Concatenate the list of Python DataFrames
    y_test_df =pd.concat(y_test_file_list)


     # train model
    params={
        'max_leaf_nodes': args.max_leaf_nodes,
        'min_samples_leaf': args.min_samples_leaf,
        'max_depth': args.max_depth,
        'n_estimators': args.n_estimators,
     
        'random_state': 11,
        'bootstrap': args.bootstrap,
        'oob_score': args.oob_score,
        }
    

    model, results = train_model(x_train_df ,x_test_df, y_train_df,y_test_df,params)

    print('Saving model...')
    mlflow.sklearn.save_model(model, args.model_output)
    
    print('Saving evauation results...')
    with open(Path(args.evaluation_report) / 'results.json', 'w') as fp:
        json.dump(results, fp)
    


def train_model(X_train ,X_test, y_train,y_test,params):
        
    # Create a Random Forest Classifier
    rf = RandomForestClassifier(**params)

    # Train the model
    model = rf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)

    # Evaluate the Model


    accuracy=accuracy_score(y_test, y_pred)
    f1=f1_score(y_test, y_pred)
    f1_micro=f1_score(y_test, y_pred, average='micro')
    f1_macro=f1_score(y_test, y_pred, average='macro')
    precision=precision_score(y_test, y_pred)
    recall=recall_score(y_test, y_pred)
    roc_auc=roc_auc_score(y_test, y_pred)
    
    print("Model Scored Successfully")

    results={}
    results["accuracy"]=accuracy
    results["f1"]=f1
    results["f1_micro"]=f1_micro
    results["f1_macro"]=f1_macro
    results["precision"]=precision
    results["recall"]=recall
    results["roc_auc"]=roc_auc
    
    print(results)
   
    mlflow.sklearn.log_model(model, "rf_model")
    mlflow.log_metric("accuracy", float(accuracy))
    mlflow.log_metric("f1", float(f1))
    mlflow.log_metric("f1_micro", float(f1_micro))
    mlflow.log_metric("f1_macro", float(f1_macro))
    mlflow.log_metric("precision", float(precision))
    mlflow.log_metric("recall", float(recall))
    mlflow.log_metric("roc_auc", float(roc_auc))

    return model, results

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--x_train", type=str, help="Path of prepped train data")
    parser.add_argument("--x_test", type=str, help="Path of prepped train data")
    parser.add_argument("--y_train", type=str, help="Path of prepped train data")
    parser.add_argument("--y_test", type=str, help="Path of prepped train data")
    parser.add_argument("--max_leaf_nodes", type=int, help="max_leaf_nodes")
    parser.add_argument("--min_samples_leaf", type=int, help="min_samples_leaf")
    parser.add_argument("--max_depth", type=int, help="max_depth")
    parser.add_argument("--n_estimators", type=int, help="n_estimators")
    parser.add_argument("--random_state", type=int, help="random_state")
    parser.add_argument("--bootstrap", type=bool, help="bootstrap")
    parser.add_argument("--oob_score", type=bool, help="oob_score")
    parser.add_argument("--model_output", type=str, help="Path of model output")
    parser.add_argument("--evaluation_report", type=str, help="Path of evaluation report")
       
    args=parser.parse_args()
    return args
# Run script

if __name__ == "__main__":
    mlflow.start_run()
    args=parse_args()
    print('Args parsed')
    main(args)

    mlflow.end_run()
    print('Training completed')