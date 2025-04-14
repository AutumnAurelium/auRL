# Much of this code is adapted from [HuggingFace TRL](https://github.com/huggingface/trl), Copyright 2025 The HuggingFace Team.
from typing import Optional
import warnings
import requests
import atexit
import time
import logging
import torch
import torch.nn as nn
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

logger = logging.getLogger(__name__)

class VLLMClient:
    """
    A wrapper around a vLLM inference server.
    """
    
    host: str
    port: int
    group_port: int
    timeout: float
    retry_interval: float
    session: requests.Session
    
    pynccl_comm: PyNcclCommunicator
    rank: int
    
    def __init__(self, host="0.0.0.0", port=8000, group_port=51216, timeout=10.0, retry_interval=2.0):
        self.host = host
        self.port = port
        self.group_port = group_port
        self.timeout = timeout
        self.retry_interval = retry_interval
        
        self.check_server()
        self.session = requests.Session()
        self.pynccl_comm, self.rank = self.init_communicator()
        
        atexit.register(self.close_communicator)
    
    def check_server(self):
        """
        Check server availability with retries on failure, within a total timeout duration. If the server is not up
        after the total timeout duration, raise a `ConnectionError`.
        """
        url = f"http://{self.host}:{self.port}/health/"
        start_time = time.time()  # Record the start time

        while True:
            try:
                response = requests.get(url)
            except requests.exceptions.RequestException as exc:
                # Check if the total timeout duration has passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= self.timeout:
                    raise ConnectionError(
                        f"The vLLM server at {self.host}:{self.port} could not be reached after {self.timeout} seconds. Make sure the server is running."
                    ) from exc
            else:
                if response.status_code == 200:
                    logger.info("Server is up!")
                    return

            # Retry logic: wait before trying again
            logger.info(f"Server is not up yet. Retrying in {self.retry_interval} seconds...")
            time.sleep(self.retry_interval)
    
    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
    ) -> list[list[int]]:
        """
        Generates model completions for the provided prompts.

        Args:
            prompts (`list[str]`):
                List of text prompts for which the model will generate completions.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate for each prompt.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. 1.0 means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.`1.0` means no truncation.
            top_k (`int`, *optional*, defaults to `-1`):
                Top-k sampling parameter. `-1` means no truncation.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate for each prompt.
            guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
                Regular expression to guide the decoding process.

        Returns:
            `list[list[int]]`:
                List of lists of token IDs representing the model-generated completions for each prompt.
        """
        url = f"http://{self.host}:{self.port}/generate/"
        response = self.session.post(
            url,
            json={
                "prompts": prompts,
                "n": n,
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_tokens": max_tokens,
                "guided_decoding_regex": guided_decoding_regex,
            },
        )
        if response.status_code == 200:
            return response.json()["completion_ids"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
    
    def update_param(self, name: str, new_weights: torch.Tensor):
        """
        Updates a specific named parameter in the model and broadcasts it to other processes.

        Args:
            name (`str`):
                Name of the layer whose weights are being updated.
            weights (`torch.Tensor`):
                Tensor containing the updated weights.
        """
        dtype, shape = str(new_weights.dtype), tuple(new_weights.shape)
        url = f"http://{self.host}:{self.port}/update_named_param/"
        response = self.session.post(url, json={"name": name, "dtype": dtype, "shape": shape})
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Broadcast the weights to the other processes
        self.pynccl_comm.broadcast(new_weights, src=self.rank, stream=torch.cuda.current_stream())
        self.pynccl_comm.group.barrier()
    
    def update_all_params(self, model: nn.Module):
        """
        Updates all parameters of the given model by calling `update_param` for each parameter in the model.

        Args:
            model (`nn.Module`):
                Model whose parameters (weights/biases) are to be updated.
        """
        for name, param in model.named_parameters():
            self.update_param(name, param.data)
    
    def reset_prefix_cache(self):
        """
        Resets the prefix cache for the model.
        """
        url = f"http://{self.host}:{self.port}/reset_prefix_cache/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
    
    def init_communicator(self):
        """
        Initializes the weight update group in a distributed setup for model synchronization.
        """
        # Get the tensor parallel size from the server
        url = f"http://{self.host}:{self.port}/get_tensor_parallel_size/"
        response = requests.get(url)
        if response.status_code == 200:
            tensor_parallel_size = response.json()["tensor_parallel_size"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        world_size = tensor_parallel_size + 1

        # Initialize weight update group
        url = f"http://{self.host}:{self.port}/init_communicator/"
        # In the server side, the host is set to 0.0.0.0
        response = self.session.post(url, json={"host": "0.0.0.0", "port": self.group_port, "world_size": world_size})
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Set up the communication group for weight broadcasting
        pg = StatelessProcessGroup.create(host=self.host, port=self.group_port, rank=self.rank, world_size=world_size)
        return PyNcclCommunicator(pg, device="cuda:0"), tensor_parallel_size

    def close_communicator(self):
        """
        Closes the weight update group and cleans up the communication group.
        """
        url = f"http://{self.host}:{self.port}/close_communicator/"

        try:
            response = self.session.post(url)
        except ConnectionError:
            # The server might be already down, so we don't need to close the communicator
            warnings.warn("Connection error shutting down vLLM communicator, is the server already down?")
        else:
            if response.status_code != 200:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")