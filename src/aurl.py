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
import wandb
from typing import Callable
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed

from utils import selective_log_softmax, is_conversational, apply_chat_template, unwrap_model_for_generation


class GRPOTrainer:
    accelerator: Accelerator
    
    model: PreTrainedModel
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

    def __init__(
        self,
        accelerator: Accelerator,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        reward_funcs: list[Callable],
        num_iterations=1,
        temperature=0.5,
        num_generations=8,
        beta=0.05,
        epsilon=0.2,
        epsilon_high=None,
        do_std_reward_scaling=True,
        generation_config: GenerationConfig = None
    ):
        self.accelerator = accelerator
        
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        
        self.device = self.model.device
        if self.ref_model.device != self.device:
            raise ValueError(f"Mismatch between policy model device ({self.device}) and ref model device ({self.ref_model.device})")

        self.num_iterations = num_iterations

        self.reward_funcs = reward_funcs

        self.temperature = temperature
        self.num_generations = num_generations
        self.beta = beta
        self.epsilon_low = epsilon
        self.epsilon_high = epsilon_high if epsilon_high else epsilon
        
        self.do_std_reward_scaling = do_std_reward_scaling
        
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
                top_p=1.0,
                top_k=50,
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

    def generate_rollouts(self, batch: list[dict]):
        for row in batch:
            if "prompt" not in row:
                raise KeyError("Dataset must include 'prompt' column.")

        prompts = [apply_chat_template(x["prompt"], self.tokenizer) for x in batch]

        # process with tokenizer, returns dict
        prompt_processed = self.tokenizer(
            text=prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )

        # Get original prompt tensors
        prompt_ids = prompt_processed["input_ids"]
        prompt_mask = prompt_processed["attention_mask"]
        
        # Repeat each prompt self.num_generations times
        prompt_ids = torch.repeat_interleave(prompt_ids, self.num_generations, dim=0)
        prompt_mask = torch.repeat_interleave(prompt_mask, self.num_generations, dim=0)

        # TODO: change to support gather-params-for-generation args
        with unwrap_model_for_generation(self.model, self.accelerator, False) as unwrapped_model:
            unwrapped_model: PreTrainedModel
            prompt_completion_ids = unwrapped_model.generate(prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config)

        # separate it out
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.tokenizer.eos_token_id

        # calculate index of the first EOS token in each sequence in the batch
        eos_idx = torch.full(
            (is_eos.size(0),),  # batch size
            is_eos.size(1),  # max seq. length
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
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._per_token_logprobs(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._per_token_logprobs(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._per_token_logprobs(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                    )

        # Decode the generated completions
        completions_text = self.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(batch[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
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
                key for key in batch[0] if key not in ["prompt", "completion"]
            ]
            reward_kwargs = {
                key: [example[key] for example in batch] for key in additional_cols
            }

            output_reward_func = reward_func(
                prompts=prompts, completions=completions, **reward_kwargs
            )

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

        # Log the metrics
        # mode = "eval" if self.control.should_evaluate else "train"

        # if mode == "train":
        #     self._total_train_tokens += (
        #         self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        #     )
        # self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        # self._metrics[mode]["completion_length"].append(completion_length)

        # Calculate mean reward per function, but only for samples where the function was applied
        # for i, reward_func in enumerate(self.reward_funcs):
            # if isinstance(
                # reward_func, nn.Module
            # ):  # Module instead of PretrainedModel for compat with compiled models
                # reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            # else:
                # reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            # mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            # self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        # self._metrics[mode]["reward"].append(rewards.mean().item())
        # self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
        #     prompts_to_log = gather_object(prompts_text)
        #     completions_to_log = gather_object(completions_text)
        #     rewards_to_log = rewards.tolist()

        #     if self.accelerator.is_main_process:
        #         if is_rich_available():
        #             print_prompt_completions_sample(
        #                 prompts_to_log,
        #                 completions_to_log,
        #                 rewards_to_log,
        #                 self.state.global_step,
        #             )
        #         if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
        #             import pandas as pd

        #             # For logging
        #             table = {
        #                 "step": [str(self.state.global_step)] * len(rewards),
        #                 "prompt": prompts_to_log,
        #                 "completion": completions_to_log,
        #                 "reward": rewards.tolist(),
        #             }
        #             df = pd.DataFrame(table)
        #             wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    def compute_loss(
        self, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._per_token_logprobs(
            self.model, input_ids, attention_mask, logits_to_keep
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
            if self.num_iterations > 1
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

        # Log the metrics
        # mode = "eval" if self.control.should_evaluate else "train"

        # if self.beta != 0.0:
        #     mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
        #     self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # is_clipped = (per_token_loss1 < per_token_loss2).float()
        # clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        # self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        return loss