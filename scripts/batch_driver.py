import os
from pathlib import Path



def init():
    global model

    # AZUREML_MODEL_DIR is an environment variable created during deployment
    # The path "model" is the name of the registered model's folder
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model")

    # load the model
    model = load_model(model_path)


import pandas as pd
from typing import List, Any, Union

def run(mini_batch: List[str]) -> Union[List[Any], pd.DataFrame]:
    results = []

    for file in mini_batch:
        # Read comma-delimited data into an array
        data = pd.read_csv(file)

        # Get the predictions
        predictions = model.predict(data)

        # Append prediction to results
        results.extend(predictions)

    return pd.DataFrame(results)
















print('Create Batch Endpoint Deployment')


endpoint_name = "MF-RanfomForest-Endpoint"


# endpoint configuration
endpoint = BatchEndpoint(
    name=endpoint_name,
    description="A batch endpoint for scoring images from the MNIST dataset.",
)

ml_client.batch_endpoints.begin_create_or_update(endpoint).result()

endpoint = ml_client.batch_endpoints.get(name=endpoint_name)
print(endpoint)

pipeline_component = ml_client.components.create_or_update(
    build_pipeline().component
)

"""
deployment = PipelineComponentBatchDeployment(
    name="Batch Deployment ML Pipeline",
    description="A sample deployment with pre and post processing done before and after inference.",
    endpoint_name=endpoint.name,
    component=pipeline_component,
    settings={"continue_on_step_failure": False, "default_compute": cluster_name},
)"""


deployment = ModelBatchDeployment(
  
    code_configuration=CodeConfiguration(
        code="./scripts",
        scoring_script="batch_driver.py"
    ),
    
)

ml_client.batch_deployments.begin_create_or_update(deployment).result()


endpoint = ml_client.batch_endpoints.get(endpoint_name)
endpoint.defaults.deployment_name = deployment.name
ml_client.batch_endpoints.begin_create_or_update(endpoint).result()



data_path="data"
dataset_name="MachineFailureData-2023"

dataset_unlabeled=ml_client.data.get(dataset_name,  label="latest", version=2)


input=Input(type=AssetTypes.URI_FOLDER, path=dataset_unlabeled.path)
# Create a job to score the data
print("Creating a job to score the data")

job = ml_client.batch_endpoints.invoke(
    endpoint_name=endpoint.name,
    deployment_name=deployment.name,
    input= input,
)

# Wait for the job to finish
print("Waiting for job to finish")
ml_client.jobs.stream(job.name)
print("Job finished.")
# Download the results of the job
print("Downloading results")
ml_client.jobs.download(name=job.name, output_name='score', download_path='./')




"""


# Prepare the dataset
data_path="data"
dataset_name="MachineFailureData-2023"




















print("Dataset now exists.")

# Retrieve the model

model=ml_client.models.get(name="FailurePredictionModel", version="3")



try:
    dataset_unlabeled=ml_client.data.get(dataset_name,  label="latest")
    print("Dataset already exists.")
except Exception:
    print("No dataset exists--creating a new dataset")
    dataset_unlabeled=Data(
        path=data_path,
        type=AssetTypes.URI_FOLDER,
        description="An unlabeled dataset for Chicago parking ticket payment status",
        name=dataset_name
    )

    ml_client.data.create_or_update(dataset_unlabeled)
    print('Dataset created.')
    dataset_unlabeled=ml_client.data.get(dataset_name, label="latest")

"""