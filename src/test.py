from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from accelerate import Accelerator
from accelerate import DeepSpeedPlugin
from torch.utils.data import DataLoader
import bitsandbytes as bnb
from tqdm.auto import tqdm
import wandb
import json
from aurl import GRPOTrainer
from rewards import letter_reward, poem_topics
import random
import os

POETRY_PROMPT = """You are a poet. Write a short, impactful poem on the subject requested."""

if __name__ == "__main__":
    epochs = 1
    batch_size = 1
    num_warmup_steps = 10
    
    adam_betas = (0.9, 0.999)
    adam_weight_decay = 0.01
    initial_lr = 1e-5
    
    clip_grad_norm = 1.0
    
    gradient_accumulation_steps = 2
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    deepspeed_plugin = DeepSpeedPlugin()
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="wandb",
        deepspeed_plugin=deepspeed_plugin
    )
    
    if accelerator.is_main_process:
        wandb.init(
            "aurelium",
            "auRL",
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "num_warmup_steps": num_warmup_steps,
                "adam_betas": adam_betas,
                "adam_weight_decay": adam_weight_decay,
                "initial_lr": initial_lr,
                "clip_grad_norm": clip_grad_norm,
                "model": model_name
            }
        )
    
    dataset = Dataset.from_list([
        {
            "prompt": json.dumps([
                {"role": "system", "content": POETRY_PROMPT},
                {"role": "user", "content": f"Please write a poem about '{random.choice(poem_topics)}'"}
            ])
        }
        for _ in range(1000)
    ])
    
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    train_dataloader = accelerator.prepare(train_dataloader)
    
    policy = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    policy = accelerator.prepare(policy)
    old_policy = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    old_policy = accelerator.prepare(old_policy)
    ref = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    ref = accelerator.prepare(ref)
    tok = AutoTokenizer.from_pretrained(model_name)
    
    trainer = GRPOTrainer(
        accelerator,
        policy,
        ref,
        tok,
        [letter_reward],
        num_iterations=2,
        beta=0.1
    )
    
    optimizer = bnb.optim.Adam8bit(
        policy.parameters(),
        lr=initial_lr,
        betas=(adam_betas[0], adam_betas[1]),
        weight_decay=adam_weight_decay,
    )
    optimizer = accelerator.prepare(optimizer)
    
    num_training_steps = epochs * len(train_dataloader) * trainer.num_iterations
    
    lr_scheduler = get_scheduler(
        name="constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)
    progress_bar = tqdm(range(num_training_steps))
    
    completion_artifact_name = f"completions_{wandb.util.generate_id()}"
        
    policy.train()
    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.main_process_first():
                old_policy.load_state_dict(policy.state_dict())
            
            # if the prompt is a JSON string, convert it to a list/dict
            # if it fails to parse, use as a normal string
            try:
                prompts = []
                for prompt in batch["prompt"]:
                    prompts.append(json.loads(prompt))
                
                batch["prompt"] = prompts
            except Exception as _:
                pass
            
            for i in range(trainer.num_iterations):
                policy.eval()
                rollouts = trainer.generate_rollouts(batch, old_model=old_policy, iteration=i)
                policy.train()
                
                with accelerator.accumulate(policy):
                    loss, metrics = trainer.compute_loss(rollouts)
                    
                    if accelerator.is_main_process:
                        completions = rollouts["metrics"]["completions"]
                        rollouts["metrics"]["completions"] = None
                        
                        # log metrics and completions
                        other_keys = [k for k in batch.keys() if k != "prompt"]
                        
                        data = []
                        
                        for completion in completions:
                            data.append([
                                step,
                                batch["prompt"][0],
                                completion
                            ] + [batch[k][0] for k in other_keys])
                        
                        artifact = wandb.Artifact(completion_artifact_name, type="table")
                        artifact.add(wandb.Table(columns=["step", "prompt", "completion"] + other_keys, data=data), "completions")
                        wandb.log(metrics, step=progress_bar.n)
                        wandb.log_artifact(artifact, name=f"completions/{progress_bar.n}")
                    
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(
                        policy.parameters(), clip_grad_norm
                    )
                    
                    optimizer.step()
                    lr_scheduler.step()
                    
                    optimizer.zero_grad()
                
                progress_bar.update()