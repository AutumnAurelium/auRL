from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

from aurl import GRPOTrainer

def gsm8k_reward(output: str, answer: str):
    pass

if __name__ == "__main__":
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        log_with="wandb"
    )
    
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    ref = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    dataset = Dataset.from_json("data/gsm8k.jsonl")
    
    trainer = GRPOTrainer(
        model,
        ref,
        tok
    )
    
    for i in range(10):
        rollouts = trainer.generate_rollouts([dataset[i]])
        loss = trainer.compute_loss(rollouts)
    
    trainer.train()