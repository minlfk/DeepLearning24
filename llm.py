import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import pipeline
from rag import *

GPT2 = 'gpt2'
MISTRAL = 'mistralai/Mistral-7B-Instruct-v0.1'
MATHSTRAL = 'mistralai/mathstral-7B-v0.1'
LLAMA3 = "meta-llama/Meta-Llama-3-8B"
T5 = "google/flan-t5-small"
DISTIL_BERT = "distilbert-base-uncased"
TINY_LLAMA = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

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
        if(model_name == DISTIL_BERT):
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
            self.model = DistilBertModel.from_pretrained(self.model_name)
        if(model_name == TINY_LLAMA):
            # messages = [
            #     {
            #         "role": "system",
            #         "content": "You are a friendly chatbot who always responds in the style of a pirate",
            #     },
            #     {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
            # ]
            self.pipe = pipeline("question-answering", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
            # prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            # print(outputs[0]["generated_text"])

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
        prompt = "Comment on the following situation given the 2 possible choices. Your response should take into account all perspectives: moral and immoral.\n"
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

        
    