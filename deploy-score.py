import sys
import os
import timeit
from datetime import datetime
import numpy as np
import pandas as pd
from random import randrange
import urllib
from urllib.parse import urlencode
from azure.ai.ml import MLClient, Input, load_component
from azure.ai.ml.entities import BatchEndpoint, ModelBatchDeployment, ModelBatchDeploymentSettings, PipelineComponentBatchDeployment, Model, AmlCompute, Data, BatchRetrySettings, CodeConfiguration, Environment, Data
from azure.ai.ml.constants import AssetTypes, BatchDeploymentOutputAction
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential
import azure.ai.ml
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.entities import (
    BatchEndpoint,
    ModelBatchDeployment,
    ModelBatchDeploymentSettings,
    Model,
    AmlCompute,
    Data,
    BatchRetrySettings,
    CodeConfiguration,
    Environment,
)
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, AzureCliCredential
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
from azure.ai.ml.constants import AssetTypes, BatchDeploymentOutputAction

from azure.ai.ml.sweep import (
    Choice,
    Uniform
)



# Set workspace & compute name
workspace_name = "FailurePrediction"
cluster_name = "MachinFailureCompute"

ml_client=MLClient.from_config(AzureCliCredential())

print("Retrieved model.")


model=ml_client.models.get(name="FailurePredictionModel", version="3")

ws=ml_client.workspaces.get(workspace_name)
try:
    cpu_cluster=ml_client.compute.get(cluster_name)
    print(
        f"You already have a cluster named {cluster_name}, we'll reuse it as is."
    )

except Exception:
    print("Creating a new cpu compute target...")

    # Let's create the Azure Machine Learning compute object with the intended parameters
    # if you run into an out of quota error, change the size to a comparable VM that is available.\
    # Learn more on https://azure.microsoft.com/en-us/pricing/details/machine-learning/.

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
    description="Custom environment",
    tags={"scikit-learn": "0.24.2"},
    conda_file="conda_dependencies.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",

)
env = ml_client.environments.create_or_update(env)


print(
    f"Environment with name {env.name} is registered to workspace, the environment version is {env.version}"
)








print('Connect Done')
parent_dir="./config"


# Load the components
prepare_score = load_component(source=os.path.join(parent_dir, "prepare-score.yml"))
score_model = load_component(source=os.path.join(parent_dir, "score-model.yml"))




@pipeline(name="training_pipeline", description="Build a training pipeline")

def build_pipeline(raw_data):
    # Takes in raw data, outputs a data that is preprocessed
    step_prepare_data=prepare_score(input_data=raw_data)
    # Output is a preprocessed dataframe
    # Takes in preprocessed data, outputs a data that is prepared
    # Outputs are dataframes x_train, x_test, y_train, y_test
    print('Ready for Scoring')
    # Set training parameters
 

    # Takes in prepared data, outputs a model
    
    train_model_data=score_model(input_data=step_prepare_data.outputs.output_data,
                                    input_model=model,                                 
                                   )
                                 

    print('Training Done, registering model')
    # Takes in model, outputs a model and a test report
    return { "Prediction": train_model_data.outputs.output_data}


def prepare_pipeline_job(cluster_name):
    # must have a dataset already in place    
    data_asset=ml_client.data.get(name="MachineFailureData-2023", version="2")
    print('Data Asset Found')
    #print(data_asset)
    #raw_data = pd.read_csv(data_asset.path)


    data_input=Input(type=AssetTypes.URI_FOLDER, path=data_asset.path)
    print('Data Asset Input')
    pipeline_job=build_pipeline(data_input)

    # set pipeline level compute
    pipeline_job.settings.default_compute=cluster_name
    # set pipeline level datastore
    pipeline_job.settings.default_datastore="workspaceblobstore"
    pipeline_job.settings.force_rerun=True
    pipeline_job.display_name="score_pipeline"
    return pipeline_job

# Submit the pipeline job

prepped_job=prepare_pipeline_job(cluster_name)
ml_client.jobs.create_or_update(prepped_job, experiment_name="Machine Failure Prediction - Random Forest Classifier")
print('Job Submitted')


