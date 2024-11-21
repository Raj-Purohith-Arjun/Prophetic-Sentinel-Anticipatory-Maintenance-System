
import sys # for sys.argv
import os # for os.environ
import timeit # for timeit.default_timer
from datetime import datetime # for datetime.now
import numpy as np
import pandas as pd
from random import randrange
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import command
from azure.ai.ml import Input, Output
from azure.ai.ml import load_component
from azure.identity import AzureCliCredential
import azure.ai.ml
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.entities import Workspace, AmlCompute
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, AzureCliCredential
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.sweep import (
    Choice,
    Uniform
)



# authenticate
credential = DefaultAzureCredential()


# Set workspace & compute name
workspace_name = "FailurePrediction"
cluster_name = "MachinFailureCompute"

# Set client & workspace
ml_client = MLClient.from_config(AzureCliCredential())
                                            
ws = ml_client.workspaces.get(workspace_name)
print('Done')
try:
    cpu_cluster = ml_client.compute.get(cluster_name)
    print(f"You already have a cluster named {cluster_name}, we'll reuse it as is.")
except Exception:
    cpu_cluster=AmlCompute(
        name=cluster_name,
        # Azure Machine Learning Compute is the on-demand VM service
        type="amlcompute",
        # VM Family
        size="STANDARD_DS3_V2",
        # Minimum running nodes when there is no job running
        min_instances=0,
        # Nodes in cluster
        max_instances=4,
        # How many seconds will the node running after the job termination
        idle_time_before_scale_down=180,
        # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
        tier="Dedicated",
    )
    
    print(
        f"AMLCompute with name {cpu_cluster.name} will be created, with compute size {cpu_cluster.size}"
    )
    # Now, we pass the object to MLClient's create_or_update method
    cpu_cluster=ml_client.compute.begin_create_or_update(cpu_cluster)

from azure.ai.ml.entities import Environment

custom_env_name = "aml-scikit-learn"

env= Environment(
    name=custom_env_name,
    description="Custom environment pipeline",
    tags={"scikit-learn": "0.24.2"},
    conda_file="conda_dependencies.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    version="0.2.0",
)
env = ml_client.environments.create_or_update(env)


print(
    f"Environment with name {env.name} is registered to workspace, the environment version is {env.version}"
)





print('Connect Done')
parent_dir="./config"


# Load components
preprocess_data = load_component(source=os.path.join(parent_dir, "preprocess-data.yml"))
prepare_data=load_component(source=os.path.join(parent_dir, "prepare-data.yml"))
train_model=load_component(source=os.path.join(parent_dir, "train-model.yml"))
register_model=load_component(source=os.path.join(parent_dir, "register-model.yml"))

print('Components Loaded')
# Let azure ML know that this is a pipeline
@pipeline(name="training_pipeline", description="Build a training pipeline")

def build_pipeline(raw_data):
    # Takes in raw data, outputs a data that is preprocessed
    step_preprocess_data=preprocess_data(input_data=raw_data)
    # Output is a preprocessed dataframe
    # Takes in preprocessed data, outputs a data that is prepared
    # Outputs are dataframes x_train, x_test, y_train, y_test
    step_prepare_data=prepare_data(input_data=step_preprocess_data.outputs.output_data)

    print('Ready for training')
    # Set training parameters
 

    # Takes in prepared data, outputs a model
    
    train_model_data=train_model(x_train=step_prepare_data.outputs.output_data_x_train,
                                   y_train=step_prepare_data.outputs.output_data_y_train,
                                      x_test=step_prepare_data.outputs.output_data_x_test,
                                        y_test=step_prepare_data.outputs.output_data_y_test,
                                             n_estimators =100,
                                                max_depth= 4,
                                                min_samples_leaf= 3,
                                                bootstrap=True,
                                                oob_score=False,
                                                random_state=42,
                                                max_leaf_nodes=10,
                                                                            
                                   )
                                 
    print(train_model_data.outputs.model_output)
    print('Training Done, registering model')
    # Takes in model, outputs a model and a test report
    register_model(model=train_model_data.outputs.model_output, evaluation_report=train_model_data.outputs.evaluation_report)
    return { "model": train_model_data.outputs.model_output,
             "report": train_model_data.outputs.evaluation_report }


from azure.ai.ml.constants import AssetTypes, InputOutputModes

def prepare_pipeline_job(cluster_name):
    # must have a dataset already in place    
    data_asset=ml_client.data.get(name="MachineFailureData-Training", version="1")
    print('Data Asset Found')
    #print(data_asset)
    #raw_data = pd.read_csv(data_asset.path)


    raw_data=Input(type=AssetTypes.URI_FOLDER, path=data_asset.id)
    print('Data Asset Input')
    pipeline_job=build_pipeline(raw_data)

    # set pipeline level compute
    pipeline_job.settings.default_compute=cluster_name
    # set pipeline level datastore
    pipeline_job.settings.default_datastore="workspaceblobstore"
    pipeline_job.settings.force_rerun=True
    pipeline_job.display_name="train_pipeline"
    return pipeline_job

# Submit the pipeline job

prepped_job=prepare_pipeline_job(cluster_name)
ml_client.jobs.create_or_update(prepped_job, experiment_name="Machine Failure Prediction - Random Forest Classifier")
print('Job Submitted')