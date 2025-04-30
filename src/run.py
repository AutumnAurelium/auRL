from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
import torch
import torch.distributed as dist

from aurl import HParams, GenerationConfig, GRPOTrainer
from rollouts.vllm_client import VLLMClient
from rewards.base64 import base64_reward, generate_dataset

from dataclasses import asdict

import wandb
from tqdm import tqdm

hparams = HParams(
    epochs=1,
    batch_size=1,
    num_warmup_steps=1,
    adam_betas=(0.9, 0.999),
    adam_weight_decay=0.01,
    initial_lr=1e-4,
    clip_grad_norm=1.0,
    gradient_accumulation_steps=1,
    generation_config=GenerationConfig(
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.0,
    ),
    num_generations=8,
    num_iterations=1,
    beta=0.5,
    epsilon_low=0.2,
    epsilon_high=0.2,
    do_std_reward_scaling=False,
)

model_name = "Qwen/Qwen3-1.7B"
use_8bit_adam = True
multigpu_method = "ddp"

def main():
    policy = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")
    ref_policy = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    vllm_client = VLLMClient(port=1111)
    
    dataset = generate_dataset(1000)
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=hparams.batch_size,
        sampler=DistributedSampler(dataset)
    )
    
    # calculate training steps and total GRPO iterations
    num_training_steps = hparams.epochs * len(train_dataloader)
    
    if use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(
            policy.parameters(),
            lr=hparams.initial_lr,
            betas=(hparams.adam_betas[0], hparams.adam_betas[1]),
            weight_decay=hparams.adam_weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=hparams.initial_lr,
            betas=(hparams.adam_betas[0], hparams.adam_betas[1]),
            weight_decay=hparams.adam_weight_decay,
        )
    
    # Note: lr_scheduler is only called once per step, not per GRPO iteration.
    lr_scheduler = get_scheduler(
        name="constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=hparams.num_warmup_steps,
        num_training_steps=num_training_steps // hparams.num_iterations,
    )
    
    trainer = GRPOTrainer(
        policy,
        ref_policy,
        tokenizer,
        hparams,
        vllm_client,
        reward_funcs=[base64_reward],
        multigpu_method=multigpu_method,
    )
    
    progress_bar = tqdm(range(num_training_steps // hparams.num_iterations))
    
    completion_artifact_name = f"completions_{wandb.util.generate_id()}"
    config = asdict(hparams)
    config["multigpu_method"] = multigpu_method
    config["use_8bit_adam"] = use_8bit_adam
    if trainer.is_main_process():
        wandb.init(
            "aurelium",
            "auRL",
            config=config
        )
    
    # start training loop
    for epoch in range(hparams.epochs):
        train_dataloader.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            if trainer.is_main_process():
                # Generate rollouts
                rollouts = trainer.generate_rollouts(batch)
                # Compute rewards
                rewards = trainer.compute_rewards(rollouts)
                # Compute logprobs for beginning of step
                step_logprobs = trainer.step_logprobs(rollouts)
                # Compute advantages
                advantages = trainer.compute_advantages(rewards)
                
                # Flatten completions
                completions = [x["completions"] for x in rollouts]
                
                # Log metrics and completions                
                data = []
                
                for completion in completions:
                    data.append([
                        step,
                        batch["prompt"][0],
                        completion
                    ])
                
                other_keys = batch.keys() - {"prompt", "answers"}
                
                artifact = wandb.Artifact(completion_artifact_name, type="table")
                artifact.add(wandb.Table(columns=["step", "prompt", "completion"] + other_keys, data=data), "completions")
                wandb.log_artifact(artifact, name=f"completions/{progress_bar.n}")
            
            # Prepare data for broadcasting
            if trainer.is_main_process():
                data_to_broadcast = [step_logprobs, advantages]
            else:
                data_to_broadcast = [None, None] 

            # Broadcast rollouts and advantages in a single list
            dist.broadcast_object_list(data_to_broadcast, src=0)
            
            # Extract objects after broadcasting
            step_logprobs, advantages = data_to_broadcast[0].to(trainer.device), data_to_broadcast[1].to(trainer.device)

            # Do GRPO iterations
            for i in range(hparams.num_iterations):
                policy.train()
                optimizer.zero_grad()
                
                loss = trainer.compute_loss(step_logprobs, advantages)
                
                if trainer.is_main_process():
                    # Use a unique step for each inner iteration
                    log_step = progress_bar.n * hparams.num_iterations + i
                    wandb.log(name="iteration_metrics", data=trainer.iteration_metrics, step=log_step)
                    
                    trainer.iteration_metrics = {}
                
                loss.backward()
                clip_grad_norm_(
                    policy.parameters(), hparams.clip_grad_norm
                )
                
                optimizer.step()
                
                trainer.sync_policy_to_vllm()
            
            if trainer.is_main_process():
                wandb.log(name="step_metrics", data=trainer.step_metrics, step=log_step)
                trainer.step_metrics = {}
            lr_scheduler.step()
            
            # update progress bar
            progress_bar.update()

if __name__ == "__main__":
    main()