# Much of this training code is adapted from [HuggingFace TRL](https://github.com/huggingface/trl), Copyright 2025 The HuggingFace Team.
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    GenerationConfig
)
import torch
import warnings
import torch.nn as nn
import deepspeed
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from typing import Callable
from contextlib import nullcontext
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from vllm_client import VLLMClient
from utils import selective_log_softmax, is_conversational, apply_chat_template, unwrap_model_for_generation, pad

class GRPOTrainer:
    accelerator: Accelerator
    
    vllm: VLLMClient
    
    policy: PreTrainedModel
    ref_policy: PreTrainedModel
    
    tokenizer: PreTrainedTokenizerBase

    num_iterations: int
    
    temperature: float
    num_generations: int
    
    beta: float
    epsilon_low: float
    epsilon_high: float

    # Whether to do the std-normalization. Some research implies disabling it is better.    
    do_std_reward_scaling: bool

    reward_funcs: list[Callable]
    
    generation_config: GenerationConfig
    
    ds3_gather_params_for_generation: bool

    def __init__(
        self,
        accelerator: Accelerator,
        vllm: VLLMClient,
        policy: PreTrainedModel,
        ref_policy: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        reward_funcs: list[Callable],
        num_iterations=1,
        temperature=0.7,
        num_generations=8,
        beta=0.05,
        epsilon=0.2,
        epsilon_high=None,
        do_std_reward_scaling=True,
        generation_config: GenerationConfig = None,
        ds3_gather_params_for_generation: bool = False
    ):
        self.accelerator = accelerator
        
        self.vllm = vllm
        
        self.policy = policy
        self.ref_policy = ref_policy
        self.tokenizer = tokenizer
        
        self.device = self.policy.device
        self.num_iterations = num_iterations

        self.reward_funcs = reward_funcs

        self.temperature = temperature
        self.num_generations = num_generations
        self.beta = beta
        self.epsilon_low = epsilon
        self.epsilon_high = epsilon_high if epsilon_high else epsilon
        
        self.do_std_reward_scaling = do_std_reward_scaling
        
        self.ds3_gather_params_for_generation = ds3_gather_params_for_generation
        
        if generation_config is not None:
            self.generation_config = generation_config
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=512,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=self.temperature,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                repetition_penalty=1.0
            )

    def _per_token_logprobs(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits
        logits = logits[
            :, :-1, :
        ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]

        # Divide logits by sampling temperature.
        # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
        logits = logits / self.temperature
        return selective_log_softmax(
            logits, input_ids
        )  # compute logprobs for the input tokens

    def generate_rollouts(self, batch: dict[str, list]):
        if "prompt" not in batch:
            raise KeyError("Dataset must include 'prompt' column.")

        metrics = {}

        prompts = [apply_chat_template(x, self.tokenizer) for x in batch["prompt"]]

        # process with tokenizer, returns dict
        prompt_processed = self.tokenizer(
            text=prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )

        # Get original prompt tensors
        prompt_ids = prompt_processed["input_ids"].to(self.device)
        prompt_mask = prompt_processed["attention_mask"].to(self.device)
        
        # Repeat each prompt self.num_generations times *locally*
        prompt_ids = torch.repeat_interleave(prompt_ids, self.num_generations, dim=0)
        prompt_mask = torch.repeat_interleave(prompt_mask, self.num_generations, dim=0)

        # Gather all prompts onto the main process for generation
        all_prompts = gather_object(prompts)
        
        all_completion_ids_list = None
        if self.accelerator.is_main_process:
            # Generate completions for all gathered prompts
            all_completion_ids_list = self.vllm.generate(
                all_prompts, # Use gathered prompts
                n=self.num_generations,
                max_tokens=512,
                temperature=self.temperature,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                repetition_penalty=1.0
            )
            
            # Chunk the flat list of completions
            # Expected structure: [[p0_g0, p0_g1,...], [p1_g0, p1_g1,...], ...]
            num_all_prompts = len(all_prompts)
            if len(all_completion_ids_list) != num_all_prompts * self.num_generations:
                raise ValueError(f"Mismatch between expected ({num_all_prompts * self.num_generations}) and actual ({len(all_completion_ids_list)}) number of completions.")
                
            chunked_completion_ids = [
                all_completion_ids_list[i * self.num_generations:(i + 1) * self.num_generations]
                for i in range(num_all_prompts)
            ]
        else:
            chunked_completion_ids = None # Placeholder for non-main processes

        # Scatter the chunked completions back to the corresponding processes
        # Each process receives a list of lists: [[local_p0_g0, local_p0_g1,...], [local_p1_g0, local_p1_g1,...], ...]
        local_chunked_completion_ids = self.accelerator.scatter_object(chunked_completion_ids)
        
        # Flatten the received list of lists to match the repeated prompt_ids structure
        local_completion_ids = [item for sublist in local_chunked_completion_ids for item in sublist]

        # Pad the completions, and concatenate them with the prompts
        completion_ids = [torch.tensor(ids, device=self.accelerator.device) for ids in local_completion_ids] # Use local_completion_ids
        completion_ids = pad(completion_ids, padding_value=self.tokenizer.pad_token_id)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.tokenizer.eos_token_id

        # calculate index of the first EOS token in each sequence in the batch
        eos_idx = torch.full(
            (is_eos.size(0),),  # batch size
            is_eos.size(1),     # max seq. length
            dtype=torch.long,
            device=self.device,
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]

        # index of every token in sequence
        sequence_indices = torch.arange(is_eos.size(1), device=self.device).expand(
            is_eos.size(0), -1
        )
        # mask where index within completion
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # (batch_size, prompt_length+completion_length) concatenated mask
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            # When using num_iterations == 1, ßper_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                with unwrap_model_for_generation(self.policy, self.accelerator, self.ds3_gather_params_for_generation) as unwrapped_model:
                    old_per_token_logps = self._per_token_logprobs(
                        unwrapped_model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_policy is not None:
                with unwrap_model_for_generation(self.ref_policy, self.accelerator, self.ds3_gather_params_for_generation) as unwrapped_model:
                    ref_per_token_logps = self._per_token_logprobs(
                        unwrapped_model, prompt_completion_ids, attention_mask, logits_to_keep
                    ).detach()
            else:
                warnings.warn("No reference policy provided, but beta is not 0. No KL divergence will be computed.")
                ref_per_token_logps = None

        # Decode the generated completions
        completions_text = self.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        
        if is_conversational(batch["prompt"][0]):
            completions = []
            for prompt, completion in zip(batch["prompt"] * len(completions_text), completions_text):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(
            len(completions), len(self.reward_funcs), device=self.device
        )
        for i, reward_func in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            additional_cols = [
                key for key in batch.keys() if key not in ["prompt", "completion"]
            ]
            reward_kwargs = {
                key: batch[key] for key in additional_cols
            }

            output_reward_func = reward_func(
                prompts=batch["prompt"], completions=completions, **reward_kwargs
            )
            
            if None in output_reward_func:
                print(f"{reward_func.__name__} returned None for the following kwargs: {reward_kwargs}. It will be treated as 0.")

            # Convert None values to NaN
            output_reward_func = [
                reward if reward is not None else torch.nan
                for reward in output_reward_func
            ]

            rewards_per_func[:, i] = torch.tensor(
                output_reward_func, dtype=torch.float32, device=self.device
            )

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = (
                torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            )
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items()
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        # TODO: change for accelerate
        rewards_per_func = gather(rewards_per_func.to(self.device))

        # Sum while treating NaNs as zeroes
        rewards = rewards_per_func.nansum(dim=1)

        # Compute group-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        
        advantages = rewards - mean_grouped_rewards
        if self.do_std_reward_scaling:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        
        if self.accelerator.is_main_process:
            # metrics!
            reward_metrics = {}
            for i in range(rewards_per_func.size(1)):
                func_rewards = rewards_per_func[:, i]
                reward_metrics[self.reward_funcs[i].__name__] = {
                    "mean": func_rewards.mean().item(),
                    "std": func_rewards.std().item(),
                    "min": func_rewards.min().item(),
                    "max": func_rewards.max().item()
                }
            
            metrics["rewards"] = reward_metrics
            
            metrics["completions"] = completions

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "metrics": metrics,
        }
        
    def sync_policy_to_vllm(self):
        # TODO: is this equivalent?
        gather_if_zero3 = deepspeed.zero.GatheredParameters if is_deepspeed_zero3_enabled() else nullcontext
        
        for name, param in self.policy.named_parameters():
            with gather_if_zero3([param]):
                if self.accelerator.is_main_process:
                    self.vllm.update_param(name, param.data)
        
        if self.accelerator.is_main_process:
            self.vllm.reset_prefix_cache()

    def compute_loss(
        self, inputs
    ):
        metrics = inputs["metrics"]

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._per_token_logprobs(
            self.policy, input_ids, attention_mask, logits_to_keep
        )

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation (see
        # _generate_rollouts) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            inputs["old_per_token_logps"]
            if self.num_iterations > 1 and inputs["old_per_token_logps"] is not None
            else per_token_logps.detach()
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        if self.accelerator.is_main_process:
            metrics["loss_stats"] = {
                "loss": loss.item(),
                "mean_advantage": advantages.mean().item(),
                "policy_ratio": {
                    "mean": coef_1.mean().item(),
                    "min": coef_1.min().item(),
                    "max": coef_1.max().item()
                }
            }
            
            if self.beta != 0.0:
                mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
                metrics["loss_stats"]["mean_kl"] = mean_kl.item()
            
            is_clipped = (per_token_loss1 < per_token_loss2).float()
            clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
            metrics["loss_stats"]["clip_ratio"] = clip_ratio.item()

        return loss, metrics