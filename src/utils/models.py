import itertools

import torch

# This function is taken directly from [HuggingFace TRL](https://github.com/huggingface/trl), Copyright 2025 The HuggingFace Team.
def get_all_parameters(sub_module: torch.nn.Module, recurse: bool = False) -> list[torch.Tensor]:
    """
    Get all parameters of a module.

    Args:
        sub_module (torch.nn.Module): The module to get the parameters of.
        recurse (bool): Whether to recurse into submodules.

    Returns:
        list[torch.Tensor]: A list of all parameters of the module.
    """
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())

# This function is taken directly from [HuggingFace TRL](https://github.com/huggingface/trl), Copyright 2025 The HuggingFace Team.
def iter_params(module: torch.nn.Module, recurse: bool = False) -> list[torch.Tensor]:
    """
    Iterate over all parameters of a module.

    Args:
        module (torch.nn.Module): The module to iterate over the parameters of.
        recurse (bool): Whether to recurse into submodules.

    Returns:
        list[torch.Tensor]: A list of all parameters of the module.
    """
    return [param for _, param in get_all_parameters(module, recurse)]