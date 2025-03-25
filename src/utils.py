import torch
from transformers import PreTrainedTokenizer
from typing import Union
from packaging import version
from contextlib import contextmanager
import deepspeed
from deepspeed import DeepSpeedEngine
from accelerate import Accelerator
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import itertools

# This function is taken directly from [HuggingFace TRL](https://github.com/huggingface/trl), Copyright 2025 The HuggingFace Team.
def selective_log_softmax(logits: torch.Tensor, index: int):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps

# TODO: is this actually sufficient
def is_conversational(prompt: any) -> bool:
    return isinstance(prompt, list)

# TODO: is this also sufficient
def apply_chat_template(prompt: any, tokenizer: PreTrainedTokenizer) -> str:
    if is_conversational(prompt):
        return tokenizer.apply_chat_template(prompt, tokenize=False)
    else:
        return prompt

# This function is taken directly from [HuggingFace TRL](https://github.com/huggingface/trl), Copyright 2025 The HuggingFace Team.
def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())

# This function is taken directly from [HuggingFace TRL](https://github.com/huggingface/trl), Copyright 2025 The HuggingFace Team.
def iter_params(module, recurse=False):
    return [param for _, param in get_all_parameters(module, recurse)]

# This function is taken directly from [HuggingFace TRL](https://github.com/huggingface/trl), Copyright 2025 The HuggingFace Team.
def remove_hooks(model: DeepSpeedEngine) -> None:
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if not hasattr(model, "optimizer"):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")

    for param in iter_params(optimizer_offload.module, recurse=True):
        param.ds_active_sub_modules.clear()

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []

# This function is adapted slightly from [HuggingFace TRL](https://github.com/huggingface/trl), Copyright 2025 The HuggingFace Team.
def add_hooks(model: DeepSpeedEngine) -> None:
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if not hasattr(model, "optimizer"):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")
    
    # This is named something different in deepspeed < 0.16.4, but we require >=0.16.4
    # https://github.com/deepspeedai/DeepSpeed/pull/6847
    optimizer_offload._register_deepspeed_module(optimizer_offload.module)

# This function is taken directly from [HuggingFace TRL](https://github.com/huggingface/trl), Copyright 2025 The HuggingFace Team.
@contextmanager
def unwrap_model_for_generation(
    model: Union[DistributedDataParallel, DeepSpeedEngine],
    accelerator: Accelerator,
    gather_deepspeed3_params: bool = True,
):
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
        if not gather_deepspeed3_params:
            yield accelerator.unwrap_model(model)
        else:
            with deepspeed.zero.GatheredParameters(model.parameters()):
                remove_hooks(model)
                yield accelerator.unwrap_model(model)
                add_hooks(model)
    else:
        yield unwrapped_model