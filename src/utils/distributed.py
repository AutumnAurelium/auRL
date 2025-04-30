from typing import Callable
from transformers import PreTrainedModel
import torch.nn as nn
from torch.distributed.fsdp import fully_shard

def fully_shard_model(model: PreTrainedModel, should_shard: Callable[[nn.Module], bool]) -> None:
    """
    Fully shards appropriate modules in the model using FSDP.
    """
    for name, module in model.named_modules():
        if should_shard(module):
            fully_shard(module)
    
    fully_shard(model)
    