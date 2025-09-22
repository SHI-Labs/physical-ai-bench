"""Utilities for writing models and managing the weights of those models.

See RAY_PIPELINES.md for an introduction to our model packaging system.
"""

import abc
import concurrent.futures
import hashlib
import json
import os
import pathlib
import socket
import typing
from typing import Any, List, Optional

import attrs
import huggingface_hub
import ray
import requests
import tqdm

# from cosmos_s3_utils import s3
from filelock import FileLock
from loguru import logger

# import runtime_config
from utils import tmp_files

T = typing.TypeVar("T")

CUDA_DEVICE = "cuda:0"

# A location that is local to the worker node, where weights will be pulled
# from staging storage or PBSS.
_LOCAL_MODEL_PATH = pathlib.Path("checkpoint")


def get_local_dir_for_weights_name(weights_name: str) -> pathlib.Path:
    """Gets the local directory for a given name for a set of weights.

    See RAY_PIPELINES.md and ModelInterface for more info.
    """
    return _LOCAL_MODEL_PATH / weights_name


class ModelInterface(abc.ABC):
    """
    Abstract base class that defines an interface for machine learning models,
    specifically focused on their weight handling and environmental setup.

    This interface allows our testing and Ray pipeline code to download weights locally and setup models in a uniform
    way. It does not place any restrictions on how inference is run.

    See RAY_PIPELINES.md for a high level overview.
    """

    @property
    @abc.abstractmethod
    def weights_names(self) -> list[str]:
        """
        Returns a list of weight names associated with the model.

        In yotta, each set of weights has a name associated with it. This is oftem the huggingspace name for those
        weights (e.g. Salesforce/instructblip-vicuna-13b). but doesn't need to be. We use these names to push/pull
        weights to/from pbss.

        Returns:
            A list of strings.
        """
        pass

    @property
    def num_gpus(self) -> float:
        """
        Returns the number of GPUs needed to run a single instance of this model.

        This will typically be 1, but may be greater than 1 for model-parallel models or less than one for models which
        only take up a small amount of a GPUs memory footprint or processing power.

        Defaults to 1.0 if not sepcified.

        Returns:
            A float representing the number of GPUs.
        """
        return 1.0

    @property
    @abc.abstractmethod
    def conda_env_name(self) -> str:
        """
        Returns the name of the conda environment that this model must be run from.

        Returns:
            A string representing the conda environment name.
        """
        pass

    @abc.abstractmethod
    def setup(self) -> None:
        """
        Sets up the model for use, such as loading weights and building computation graphs.
        """
        pass

    def _generate_lock_file_name(self) -> pathlib.Path:
        """Generate a unique lock file name based on list of weights"""
        combined_weights = "".join(sorted(self.weights_names))
        lock_name = hashlib.sha256(combined_weights.encode()).hexdigest()
        return pathlib.Path(f"/tmp/{lock_name}.lock")

    def _check_if_weights_exist(self, location_prefix: pathlib.Path) -> List[str]:
        """Util function to check if weights exist in the expected location.
        Returns a list of missing weights.
        """
        missing_models = []
        for model_name in self.weights_names:
            model_path = location_prefix / model_name
            if not model_path.exists():
                missing_models.append(model_name)
            elif model_path.is_file() and model_path.stat().st_size == 0:
                missing_models.append(model_name)
            elif model_path.is_dir() and not any(
                model_path.iterdir()
            ):  # Checks if directory is empty
                missing_models.append(model_name)
        return missing_models

    def download_weights(self) -> None:
        local_location = _LOCAL_MODEL_PATH
        logger.info("Checking if weights exist locally before downloading them.")
        list_missing_weights = self._check_if_weights_exist(local_location)

        if not list_missing_weights:
            logger.info(f"Weights exist under {local_location}, skipping download.")
            return
        else:
            logger.info(
                f"The following weights were not found, downloading from local path: {list_missing_weights}"
            )


@ray.remote
def _pull_model_weights_from_ray_object_store(
    model_weight_refs: dict[str, dict[str, Any]],
) -> None:
    for model_name, weights in model_weight_refs.items():
        logger.info(
            f"Pulling {model_name} weights from Ray object store to {socket.gethostname()}"
        )
        for filepath, data_ref in weights.items():
            # create intermediate directories if they don't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            if not os.path.exists(filepath):
                with open(filepath, "wb") as fp:
                    fp.write(ray.get(data_ref))
    logger.info(f"Finished pulling weights on {socket.gethostname()}")


def pull_model_weights_from_ray_object_store_to_all_nodes(
    model_weight_refs: dict[str, dict[str, Any]],
) -> None:
    node_ips = ray.nodes()
    logger.info(f"Pulling weights on {len(node_ips)} nodes")
    bundles = [{"CPU": 1.0} for _ in range(len(node_ips))]
    pg = ray.util.placement_group(bundles=bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    futures = [
        _pull_model_weights_from_ray_object_store.options(placement_group=pg).remote(
            model_weight_refs
        )
        for _ in range(len(node_ips))
    ]
    _ = ray.get(futures)


def push_model_weights_to_ray_object_store() -> dict[str, dict[str, Any]]:
    logger.info("Pushing model weights to Ray object store")
    all_model_weights = {}
    if not pathlib.Path(_LOCAL_MODEL_PATH).exists():
        return all_model_weights
    for model_name in [
        x for x in pathlib.Path(_LOCAL_MODEL_PATH).iterdir() if x.is_dir()
    ]:
        all_model_weights[model_name] = {}
        for item in pathlib.Path(_LOCAL_MODEL_PATH, model_name).rglob("*"):
            if item.is_file():
                all_model_weights[model_name][str(item)] = ray.put(item.read_bytes())
    return all_model_weights
