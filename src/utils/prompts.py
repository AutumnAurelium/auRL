import json

from transformers import PreTrainedTokenizer

def is_conversational(prompt: any) -> bool:
    """
    Check if a prompt is conversational.
    Currently only checks if the prompt is not a string.

    Args:
        prompt (any): The prompt to check.

    Returns:
        bool: True if the prompt is not a string, False otherwise.
    """
    return not isinstance(prompt, str)

def apply_template(prompt: any, tokenizer: PreTrainedTokenizer) -> str:
    """
    If a prompt is conversational, apply the template to it.
    Otherwise, return the prompt as-is.

    Args:
        prompt (any): The prompt to apply the template to.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.

    Returns:
        str: The prompt with the template applied if it is conversational, otherwise the prompt as-is.
    """
    if is_conversational(prompt):
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    else:
        return prompt
