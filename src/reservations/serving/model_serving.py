import os
import mlflow
import requests
from databricks.sdk import WorkspaceClient
from src.reservations.config import databricks_config
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
    print(f"Latest model version: {latest_version}")
    return latest_version

def deploy_or_update_serving_endpoint(model_name, endpoint_name, version="latest", workload_size="Small", scale_to_zero=True):
    workspace = WorkspaceClient()
    endpoint_exists = any(item.name == endpoint_name for item in workspace.serving_endpoints.list())

    if version == "latest":
        entity_version = get_latest_model_version(model_name)
    else:
        entity_version = version

    served_entities = [
        ServedEntityInput(
            entity_name=model_name,
            scale_to_zero_enabled=scale_to_zero,
            workload_size=workload_size,
            entity_version=entity_version,
        )
    ]

    if not endpoint_exists:
        workspace.serving_endpoints.create(
            name=endpoint_name,
            config=EndpointCoreConfigInput(served_entities=served_entities),
        )
    else:
        workspace.serving_endpoints.update_config(name=endpoint_name, served_entities=served_entities)

def call_endpoint(endpoint_name, record):
    os.environ["DBR_HOST"] = databricks_config["host"]
    os.environ["DBR_TOKEN"] = databricks_config["token"]
    serving_endpoint = f"{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"
    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
        verify=False
    )
    return response.status_code, response.text
