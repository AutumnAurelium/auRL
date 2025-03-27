from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from accelerate import Accelerator
from torch.utils.data import DataLoader
import bitsandbytes as bnb
from tqdm.auto import tqdm
import wandb
import json
from aurl import GRPOTrainer
import re

def gsm8k_reward(prompts: list[str], completions: list[str], answer: str):
    rewards = []
    for completion in completions:
        text = completion[-1]["content"]
        
        print(answer in text)
        
        no_whitespace = re.sub(r"\s+", "", text)
        
        reward = 0.0
        if f"\\boxed{{{answer}}}" in no_whitespace:
            print("she did it!")
            reward = 1.0
        elif str(answer) in no_whitespace:
            reward = 0.5
        elif re.search(r"^\\boxed{.*}$", no_whitespace):
            reward = 0.5
        
        rewards.append(reward)
    
    return rewards

GSM8K_SYSPROMPT = """The following is a grade-school math problem. Reason through it step-by-step and solve it.
When you're done, return your answer in the following format:
\\boxed{YOUR ANSWER GOES HERE}
If you do not answer in the above format, the question will be marked as incorrect."""

if __name__ == "__main__":
    epochs = 1
    batch_size = 1
    num_warmup_steps = 10
    
    adam_betas = (0.9, 0.98)
    adam_weight_decay = 0.01
    initial_lr = 3e-6
    
    clip_grad_norm = 0.01
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
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
    
    policy = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    old_policy = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    ref = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    tok = AutoTokenizer.from_pretrained(model_name)
    
    dataset = load_dataset("json", data_files="data/gsm8k.jsonl")["train"].map(lambda x: {
        "prompt": json.dumps([
            {"role": "system", "content": GSM8K_SYSPROMPT},
            {"role": "user", "content": x["prompt"]}
        ]),
        "answer": x["answer"]
    })
    
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    trainer = GRPOTrainer(
        accelerator,
        policy,
        ref,
        tok,
        [gsm8k_reward],
        num_iterations=2
    )
    
    optimizer = bnb.optim.Adam8bit(
        policy.parameters(),
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