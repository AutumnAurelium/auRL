# Much of this training code is adapted from [HuggingFace TRL](https://github.com/huggingface/trl), Copyright 2025 The HuggingFace Team.
import torch.distributed.tensor
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel
)
import torch
from typing import Callable, Literal

from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from rollouts.vllm_client import VLLMClient

from utils.tensors import selective_log_softmax, pad
from utils.prompts import apply_template
from utils.distributed import fully_shard_model

from typing import TypedDict

from dataclasses import dataclass

import os
import warnings
class GroupRollouts(TypedDict):
    prompt: str  # The prompt used to generate the completions.
    
    completions_ids: list[list[int]]  # The generated completion token IDs.
    completions: list[str]  # The generated completions, decoded.
    
    metadata: dict  # Extra data to store for reward functions.

class StepLogprobs(TypedDict):
    prompt_ids: torch.Tensor  # The prompt token IDs.
    prompt_masks: torch.Tensor  # The attention masks for the prompt.
    
    completion_ids: torch.Tensor  # The generated completion token IDs.
    completion_masks: torch.Tensor  # The attention masks for the generated completion.
    
    old_per_token_logps: torch.Tensor | None  # The logprobs of the old policy.
    ref_per_token_logps: torch.Tensor | None  # The logprobs of the reference policy.

class StepMetrics(TypedDict):
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float
    rewards_by_func: dict[str, dict[str, float]]
    
    advantage_mean: float
    advantage_std: float

class IterationMetrics(TypedDict):
    loss: float
    policy_ratio: dict[str, float]
    mean_kl: float
    clip_ratio: float

@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    repetition_penalty: float

@dataclass
class HParams:
    epochs: int
    batch_size: int
    num_warmup_steps: int
    
    # optimizer params
    adam_betas: tuple[float, float]
    adam_weight_decay: float
    
    # training params
    initial_lr: float
    clip_grad_norm: float
    gradient_accumulation_steps: int
    
    # model
    generation_config: GenerationConfig
    
    # GRPO params
    num_generations: int
    num_iterations: int
    beta: float
    epsilon_low: float
    epsilon_high: float
    do_std_reward_scaling: bool

class GRPOTrainer:    
    local_rank: int
    global_rank: int
    device: torch.device
    world_size: int
    
    hparams: HParams
    
    vllm: VLLMClient  # The vLLM client for generating rollouts and updating parameters.
    multigpu_method: Literal["ddp", "fsdp", "none"]
    
    policy: PreTrainedModel  # The policy model we are training.
    ref_policy: PreTrainedModel  # The reference model for KL divergence computation.
    
    tokenizer: PreTrainedTokenizerBase  # The tokenizer for the model.

    reward_funcs: list[Callable]  # A list of reward functions.
    
    step_metrics: StepMetrics  # A dictionary to store metrics for the overall step.
    iteration_metrics: IterationMetrics  # A dictionary to store metrics for the GRPO iteration.

    def __init__(
        self,
        policy: PreTrainedModel,
        ref_policy: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        hparams: HParams,
        vllm: VLLMClient,
        reward_funcs: list[Callable],
        multigpu_method: Literal["ddp", "fsdp", "none"],
        custom_nosplit_modules: list[str] | None = None,
        device: torch.device | str | None = None,
    ):
        """
        Initialize the GRPOTrainer.
        """
        self.multigpu_method = multigpu_method
        if multigpu_method != "none":
            try:
                self.local_rank = int(os.environ["LOCAL_RANK"])
                self.global_rank = int(os.environ["RANK"])
                self.world_size = int(os.environ["WORLD_SIZE"])
            except KeyError:
                raise ValueError("Environment variables LOCAL_RANK, RANK, and WORLD_SIZE must be set for multi-GPU training.")
        else:
            self.local_rank = -1
            self.global_rank = -1
            self.world_size = -1
        
        if device is None:
            if self.multigpu_method == "none":
                self.device = torch.device("cuda")
            else:
                self.device = torch.device(f"cuda:{self.local_rank}")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        policy.to(self.device)
        ref_policy.to(self.device)

        self.hparams = hparams
        
        self.vllm = vllm
            
        if multigpu_method == "ddp":
            self.policy = DDP(policy, device_ids=[self.local_rank])
            self.ref_policy = DDP(ref_policy, device_ids=[self.local_rank])
        elif multigpu_method == "fsdp":
            if hasattr(policy, "_no_split_modules") and (policy._no_split_modules is None or len(policy._no_split_modules) == 0):
                if custom_nosplit_modules is None or len(custom_nosplit_modules) == 0:
                    raise ValueError("No modules to wrap with FSDP - either use a model with policy._no_split_modules set, or pass a list of module names via `custom_nosplit_modules` to wrap with FSDP.")
                else:
                    nosplit = custom_nosplit_modules
            else:
                nosplit = policy._no_split_modules
            
            def is_nosplit(module):
                return module.__class__.__name__ in nosplit

            self.policy = policy
            self.ref_policy = ref_policy
            fully_shard_model(self.policy, is_nosplit)
            fully_shard_model(self.ref_policy, is_nosplit)
        elif multigpu_method == "none":
            self.policy = policy
            self.ref_policy = ref_policy
        else:
            raise ValueError(f"Invalid multigpu_method: {multigpu_method}")

        self.ref_policy.eval()
        
        self.tokenizer = tokenizer
        
        self.device = self.policy.device

        self.reward_funcs = reward_funcs
        
        self.step_metrics = {}
        self.iteration_metrics = {}
    
    def is_main_process(self) -> bool:
        return self.multigpu_method == "none" or self.global_rank == 0

    def _per_token_logprobs(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        """
        Compute the log probabilities of a sequence of tokens.
        Assumes that the model is already in an appropriate state for generation.

        Args:
            model (PreTrainedModel): The model to compute the logprobs of the input tokens.
            input_ids (torch.Tensor): The tokens to generate logprobs for.
            attention_mask (torch.Tensor): The attention mask for the input tokens.
            logits_to_keep (int): How many logits to return.

        Returns:
            torch.Tensor: The log probabilities for the input tokens.
        """
        # We add 1 to `logits_to_keep` because the last set of logits for the sequence is later excluded
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits
        
        # Exclude the last logit, since it corresponds to the next token prediction.
        logits = logits[
            :, :-1, :
        ]

        input_ids = input_ids[:, -logits_to_keep:]

        # Divide logits by sampling temperature.
        logits = logits / self.hparams.generation_config.temperature
        return selective_log_softmax(
            logits, input_ids
        )

    def generate_rollouts(self, batch: dict[str, list]) -> list[GroupRollouts]:
        """
        Generate a full group of rollouts for each of a given set of prompts.
        
        Args:
            batch (dict[str, list]): A batch from the dataset. Must include a "prompt" column, but can include other columns for use in reward functions.

        Returns:
            list[GroupRollouts]: A list of dictionaries containing the generated token IDs and decoded completions for a given group.
        """
        if not self.is_main_process():
            raise ValueError("This function should only be called on the main process.")
        
        if "prompt" not in batch:
            raise KeyError("Dataset must include 'prompt' column.")
        
        prompts = [apply_template(x, self.tokenizer) for x in batch["prompt"]]
        metadata = []
        
        # Reformat batch to list of dicts rather than dict of lists and use as metadata
        for i in range(len(batch["prompt"])):
            prompt_metadata = {}
            for key in batch.keys():
                # Exclude "prompt" from the metadata
                if key == "prompt":
                    continue
                
                try:
                    prompt_metadata[key] = batch[key][i]
                except IndexError:
                    raise IndexError(f"All columns must exist for all prompts. Missing column: {key}.")
            metadata.append(prompt_metadata)
        
        # Generate self.hparams.num_generations completions for each prompt
        completions_ids = self.vllm.generate(
            prompts,
            n=self.hparams.num_generations,
            max_tokens=self.hparams.generation_config.max_new_tokens,
            temperature=self.hparams.generation_config.temperature,
            top_p=self.hparams.generation_config.top_p,
            top_k=self.hparams.generation_config.top_k,
            min_p=self.hparams.generation_config.min_p,
            repetition_penalty=self.hparams.generation_config.repetition_penalty
        )
        
        if len(completions_ids) != len(prompts) * self.hparams.num_generations:
            raise ValueError(f"Mismatch between expected ({len(prompts) * self.hparams.num_generations}) and actual ({len(completions_ids)}) number of completions.")
        
        groups: list[GroupRollouts] = []
        for i in range(len(prompts)):
            offset = i * self.hparams.num_generations
            group_completions_ids = completions_ids[offset:offset + self.hparams.num_generations]
            group_completions = self.tokenizer.batch_decode(group_completions_ids)
            
            groups.append({
                "prompt": prompts[i],
                "completions_ids": group_completions_ids,
                "completions": group_completions,
                "metadata": metadata[i]
            })
        
        return groups
        
    def _group_rewards(self, group: GroupRollouts) -> dict[str, list[float]]:
        """
        Grade the rollouts for a group of completions.
        
        Args:
            group (GroupRollouts): A group of completions to grade.

        Returns:
            dict[str, list[float]]: A dictionary mapping reward function names to lists of rewards for each completion.
        """
        rewards = {}
        for reward_func in self.reward_funcs:
            # Pass completions in group to reward function with prompt, completion, and metadata
            fn_rewards = reward_func(prompt=group["prompt"], completions=group["completions"], **group["metadata"])
            
            # Verify length
            if len(fn_rewards) != len(group["completions"]):
                raise ValueError(f"Reward function {reward_func.__name__} returned a list of length {len(fn_rewards)}, but expected {len(group['completions'])}.")
            
            # Verify types
            for val in fn_rewards:
                if not (isinstance(val, float) or isinstance(val, int)):
                    raise ValueError(f"Reward function {reward_func.__name__} returned a non-number value: {val}")
            
            rewards[reward_func.__name__] = fn_rewards
        
        return rewards
        
    def compute_rewards(self, groups: list[GroupRollouts]) -> torch.Tensor:
        """
        Compute the rewards for a list of groups.
        
        Args:
            groups (list[GroupRollouts]): A list of groups of completions to compute rewards for.

        Returns:
            torch.Tensor: A (batch_size, num_generations) tensor of rewards for the groups of completions.
        """
        grouped_rewards_by_func = [self._group_rewards(group) for group in groups]
        
        rewards = []
        for rewards_by_func in grouped_rewards_by_func:
            summed = [0 for _ in range(len(groups[0]["completions"]))]
            
            # Sum all rewards per-completion
            for func_name, func_rewards in rewards_by_func.items():
                summed = [summed[i] + func_rewards[i] for i in range(len(summed))]
            
            rewards.append(torch.tensor(summed))
        
        rewards_tensor = torch.stack(rewards)
        
        if self.is_main_process():
            self.step_metrics["reward_mean"] = rewards_tensor.mean().item()
            self.step_metrics["reward_std"] = rewards_tensor.std().item()
            self.step_metrics["reward_min"] = rewards_tensor.min().item()
            self.step_metrics["reward_max"] = rewards_tensor.max().item()
            
            combined = {}
            for grouped_rewards in grouped_rewards_by_func:
                for func_name, fn_rewards in grouped_rewards.items():
                    if func_name not in combined:
                        combined[func_name] = []
                    combined[func_name].extend(fn_rewards)
            
            rewards_by_func = {}
            
            for func_name, fn_rewards in combined.items():
                fn_rewards_tensor = torch.tensor(fn_rewards)
                rewards_by_func[func_name] = {
                    "mean": fn_rewards_tensor.mean().item(),
                    "std": fn_rewards_tensor.std().item(),
                    "min": fn_rewards_tensor.min().item(),
                    "max": fn_rewards_tensor.max().item()
                }
            
            self.step_metrics["rewards_by_func"] = rewards_by_func
    
        return rewards_tensor
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Calculate the advantages for a given set of rewards.
        
        Args:
            rewards (torch.Tensor): A (batch_size, num_generations) tensor of rewards for the groups of completions.

        Returns:
            torch.Tensor: A (batch_size, num_generations) tensor of advantages for the groups of completions.
        """
        
        # Compute group-wise rewards
        rewards = rewards.view(-1, self.hparams.num_generations)
        
        mean_grouped_rewards = rewards.mean(dim=1)
        std_grouped_rewards = rewards.std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.unsqueeze(1).repeat_interleave(
            self.hparams.num_generations, dim=1
        )
        std_grouped_rewards = std_grouped_rewards.unsqueeze(1).repeat_interleave(
            self.hparams.num_generations, dim=1
        )
        
        advantages = rewards - mean_grouped_rewards
        
        # Do std-scaling if enabled
        if self.hparams.do_std_reward_scaling:
            advantages = advantages / (std_grouped_rewards + 1e-4)
        
        if self.is_main_process():
            self.step_metrics["advantage_mean"] = advantages.mean().item()
            self.step_metrics["advantage_std"] = advantages.std().item()
        
        return advantages.flatten().unsqueeze(-1)
    
    def step_logprobs(self, groups: list[GroupRollouts]) -> StepLogprobs:
        """
        Generate the "old" policy logprobs with the current policy, and the reference policy logprobs with the reference policy.

        If the reference policy logprobs are not required (beta=0.0), None will be returned instead.
        If the old policy logprobs are not required (num_iterations=1), None will be returned instead.

        Args:
            groups (list[GroupRollouts]): The rollouts for this step.

        Returns:
            StepLogprobs: A dictionary containing the "old" logprobs and reference logprobs, as well as the prompt and completion token IDs and masks.
        """
        
        prompts = [group["prompt"] for group in groups]
        
        # Tokenize prompts
        prompt_processed = self.tokenizer(
            text=prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        
        prompt_ids = prompt_processed["input_ids"].to(self.device)
        prompt_masks = prompt_processed["attention_mask"].to(self.device)
        
        # Repeat each prompt self.hparams.num_generations times
        prompt_ids = torch.repeat_interleave(prompt_ids, self.hparams.num_generations, dim=0)
        prompt_masks = torch.repeat_interleave(prompt_masks, self.hparams.num_generations, dim=0)
        
        completions_ids_list = []
        for group in groups:
            completions_ids_list.extend([torch.tensor(completion_ids, device=self.device) for completion_ids in group["completions_ids"]])
        
        # Pad and concatenate completions - shape: (batch_size * num_generations, max_completion_length)
        completion_ids = pad(completions_ids_list, padding_value=self.tokenizer.eos_token_id)

        is_eos = completion_ids == self.tokenizer.eos_token_id

        # index of every token in sequence
        sequence_indices = torch.arange(is_eos.size(1), device=self.device).expand(
            is_eos.size(0), -1
        )
        
        # calculate index of the first EOS token in each sequence in the batch
        eos_idx = torch.full(
            (is_eos.size(0),),  # batch size
            is_eos.size(1),     # max seq. length
            dtype=torch.long,
            device=self.device,
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        
        # Mask everything after the first EOS token
        completion_masks = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # Concatenate prompts and completions
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        prompt_completion_masks = torch.cat([prompt_masks, completion_masks], dim=1)
        
        logits_to_keep = completion_ids.size(1)
        
        with torch.no_grad():
            # When using num_iterations == 1, per_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.hparams.num_iterations > 1:
                old_per_token_logps = self._per_token_logprobs(
                    self.policy, prompt_completion_ids, prompt_completion_masks, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.hparams.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_policy is not None:
                ref_per_token_logps = self._per_token_logprobs(
                    self.ref_policy, prompt_completion_ids, prompt_completion_masks, logits_to_keep
                ).detach()
            else:
                raise ValueError("No reference policy provided, but beta is not 0.")
        
        return {
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "prompt_ids": prompt_ids,
            "prompt_masks": prompt_masks,
            "completion_ids": completion_ids,
            "completion_masks": completion_masks
        }
        
    def sync_policy_to_vllm(self):
        if self.is_main_process():
            # If sharded, pull each parameter from wherever it is, sync to vLLM, and discard.
            for name, param in self.policy.named_parameters():
                if isinstance(param, DTensor):
                    local_shard = param.full_tensor()
                    self.vllm.update_param(name, local_shard.data)
                else:
                    self.vllm.update_param(name, param.data)
        
            self.vllm.reset_prefix_cache()

    def compute_loss(
        self, step_logprobs: StepLogprobs, advantages: torch.Tensor
    ) -> torch.Tensor:
        prompt_ids = step_logprobs["prompt_ids"]
        prompt_masks = step_logprobs["prompt_masks"]
        completion_ids = step_logprobs["completion_ids"]
        completion_masks = step_logprobs["completion_masks"]
        
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        prompt_completion_masks = torch.cat([prompt_masks, completion_masks], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Generate logprobs for the current policy
        per_token_logps = self._per_token_logprobs(
            self.policy, prompt_completion_ids, prompt_completion_masks, logits_to_keep
        )

        # Compute the KL divergence between the model and the reference model
        if self.hparams.beta != 0.0:
            ref_per_token_logps = step_logprobs["ref_per_token_logps"]
            
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
        
        # When using num_iterations == 1, old_per_token_logps == per_token_logps,
        # so we can skip generating the old logprobs and just use per_token_logps.detach()
        old_per_token_logps = (
            step_logprobs["old_per_token_logps"]
            if self.hparams.num_iterations > 1 and step_logprobs["old_per_token_logps"] is not None
            else per_token_logps.detach()
        )
        policy_ratio = torch.exp(per_token_logps - old_per_token_logps)
        clipped_policy_ratio = torch.clamp(policy_ratio, 1 - self.hparams.epsilon_low, 1 + self.hparams.epsilon_high)
    
        per_token_loss_unclipped = policy_ratio * advantages
        per_token_loss_clipped = clipped_policy_ratio * advantages
        per_token_loss = -torch.min(per_token_loss_unclipped, per_token_loss_clipped)
    
        # Add KL divergence if enabled
        if self.hparams.beta != 0.0:
            per_token_loss = per_token_loss + self.hparams.beta * per_token_kl
        
        # Loss is the mean of per-token losses, with non-completion tokens masked out
        loss = (per_token_loss * completion_masks).sum() / (completion_masks.sum() + 1e-4)

        # Collect metrics
        if self.is_main_process():
            self.iteration_metrics = {
                "loss": loss.item(),
                "policy_ratio": {
                    "mean": policy_ratio.mean().item(),
                    "min": policy_ratio.min().item(),
                    "max": policy_ratio.max().item()
                }
            }
            
            if self.hparams.beta != 0.0:
                mean_kl = (per_token_kl * completion_masks).sum() / completion_masks.sum()
                self.iteration_metrics["mean_kl"] = mean_kl.item()
            
            is_clipped = (per_token_loss_unclipped > per_token_loss_clipped).float()
            clip_ratio = (is_clipped * completion_masks).sum() / completion_masks.sum()
            self.iteration_metrics["clip_ratio"] = clip_ratio.item()

        return loss