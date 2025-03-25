from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from accelerate import Accelerator
from torch.utils.data import DataLoader
import bitsandbytes as bnb

from aurl import GRPOTrainer

def gsm8k_reward(prompts: list[str], completions: list[str], answer: str):
    if f"<answer>{answer}</answer>" in completions[0]:
        return 1.0
    else:
        return 0.0

if __name__ == "__main__":
    epochs = 1
    batch_size = 1
    num_warmup_steps = 1
    
    adam_betas = (0.9, 0.98)
    adam_weight_decay = 0.01
    initial_lr = 3e-6
    
    clip_grad_norm = 0.01
    
    
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        log_with="wandb"
    )
    
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    ref = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    dataset = load_dataset("json", data_files="data/gsm8k.jsonl")["train"]
    
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    trainer = GRPOTrainer(
        accelerator,
        model,
        ref,
        tok,
        [gsm8k_reward]
    )
    
    optimizer = bnb.optim.Adam8bit(
        model.parameters(),
        lr=initial_lr,
        betas=(adam_betas[0], adam_betas[1]),
        weight_decay=adam_weight_decay,
    )
    
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(train_dataloader) * epochs,
    )
    
    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(dataset):
            model.eval()
            print(batch)
            rollouts = trainer.generate_rollouts([batch])
            model.train()
            
            with accelerator.accumulate(model):
                loss = trainer.compute_loss(rollouts)
                
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(
                    model.parameters(), clip_grad_norm
                )
                
                optimizer.step()
                lr_scheduler.step()
                
                optimizer.zero_grad()