from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from accelerate import Accelerator
from torch.utils.data import DataLoader
import bitsandbytes as bnb
from tqdm.auto import tqdm
import wandb

from aurl import GRPOTrainer

def gsm8k_reward(prompts: list[str], completions: list[str], answer: str):
    rewards = []
    for completion in completions:
        reward = 0.0
        for c in completion:
            if c.isalpha() and c.upper() == c:
                reward += 1.0
            
            if c == "!":
                reward += 2.0
        
        rewards.append(reward / len(completion))
    
    return rewards

if __name__ == "__main__":
    epochs = 1
    batch_size = 1
    num_warmup_steps = 10
    
    adam_betas = (0.9, 0.98)
    adam_weight_decay = 0.01
    initial_lr = 3e-5
    
    clip_grad_norm = 0.01
    
    model_name = "Qwen/Qwen2.5-0.5B"
    
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        log_with="wandb"
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
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    ref = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    tok = AutoTokenizer.from_pretrained(model_name)
    
    dataset = load_dataset("json", data_files="data/gsm8k.jsonl")["train"].map(lambda x: {
        "prompt": "Tell a story while using as many capital letters and exclamation points as possible:\n\n",
        "answer": x["answer"]
    })
    
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    trainer = GRPOTrainer(
        accelerator,
        model,
        ref,
        tok,
        [gsm8k_reward],
        num_iterations=4
    )
    
    optimizer = bnb.optim.Adam8bit(
        model.parameters(),
        lr=initial_lr,
        betas=(adam_betas[0], adam_betas[1]),
        weight_decay=adam_weight_decay,
    )
    
    num_training_steps = epochs * len(train_dataloader) * trainer.num_iterations
    
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    progress_bar = tqdm(range(num_training_steps))
    
    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            old_logps = None
            for i in range(trainer.num_iterations):
                model.eval()
                rollouts = trainer.generate_rollouts(batch, generate_old_logps=(i == 0))
                model.train()
                
                if i == 0:
                    old_logps = rollouts["old_per_token_logps"]
                else:
                    rollouts["old_per_token_logps"] = old_logps
                
                with accelerator.accumulate(model):
                    loss, metrics = trainer.compute_loss(rollouts)
                    
                    if accelerator.is_main_process:
                        wandb.log(metrics)
                    
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(
                        model.parameters(), clip_grad_norm
                    )
                    
                    optimizer.step()
                    lr_scheduler.step()
                    
                    optimizer.zero_grad()
                
                progress_bar.update()