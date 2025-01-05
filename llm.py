import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from rag import *

GPT2 = 'gpt2'
MISTRAL = 'mistralai/Mistral-7B-Instruct-v0.1'
MATHSTRAL = 'mistralai/mathstral-7B-v0.1'
LLAMA3 = "meta-llama/Meta-Llama-3-8B"
T5 = "google/flan-t5-small"

class llm():
    def __init__(self, model_name = GPT2):
        super().__init__()
        self.model_name = model_name

        # Initialize LLM
        if(model_name == GPT2):
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        if(model_name == T5):
            self.tokenizer = AutoTokenizer.from_pretrained(T5, torch_dtype="auto")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(T5, torch_dtype="auto")
        if(model_name == MISTRAL or model_name == LLAMA3):
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, torch_dtype="auto")
    
    def generate(self, prompt):
    # Ensure pad_token_id is set for GPT2
        input_ids_pristine = self.tokenizer.encode(prompt, return_tensors='pt')
        output_pristine_ids = self.model.generate(
            input_ids_pristine,
            max_length=150,  # or any desired max length
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id 
        )
        output_pristine = self.tokenizer.decode(output_pristine_ids[0], skip_special_tokens=True)
        return output_pristine
    
    def generate_pluralistic(self, situation, passages):
        prompt = "Comment on the following situation using the passages provided. Your response should be balanced, pluralistic, and take into account all perspectives fairly.\n"
        prompt += "SITUATION: " + situation + "\n"
        prompt += "PASSAGES: " + "\n".join(passages)  
        return self.generate(prompt)
    
    def generate_dummy_pluralistic(self, situation):
        prompt = "Comment on the following situation. Consider diverse perspectives and provide a balanced, pluralistic response that respects different viewpoints.\n"
        prompt += "SITUATION: " + situation + "\n"
        return self.generate(prompt)
    
    def generate_vanilla(self, situation):
        prompt = "Comment on the following situation"
        prompt += "SITUATION: " + situation + "\n"
        return self.generate(prompt)

        
    