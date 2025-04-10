import re
import difflib
import base64
import random
import string

def decode_format(response: str) -> str | None:
    pattern = re.compile(
        r"<\s*thinking\s*>.*?<\s*/\s*thinking\s*>"  # Match <thinking>...</thinking>
        r"\s*"  # Allow whitespace between tags
        r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>",  # Match <answer>...</answer> and capture content
        re.DOTALL | re.IGNORECASE  # Allow . to match newlines and ignore case for tags
    )
    match = pattern.search(response)
    if match:
        return match.group(1).strip()
    else:
        return None

def base64_reward(prompts: list[str], completions: list[str], answers: list[str]):
    rewards = []
    for completion, correct_answer in zip(completions, answers):
        response = completion[-1]["content"]
        decoded_answer = decode_format(response)
        if decoded_answer:
            # Calculate similarity ratio between decoded answer and correct answer
            similarity = difflib.SequenceMatcher(None, decoded_answer, correct_answer).ratio()
            rewards.append(similarity)
        else:
            # zero reward for invalid format
            rewards.append(0.0) 
            
    return rewards

def generate_encoded_strings(length: int = 10, num_iterations: int = 1) -> str:
    """Generates a random string and returns an """
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    encoded_bytes = random_string.encode('utf-8')
    
    for _ in range(num_iterations):
        encoded_bytes = base64.b64encode(encoded_bytes)
    
    return encoded_bytes.decode('utf-8')

