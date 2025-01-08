import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import pipeline
from rag import *

# Global variables for models
GPT2 = 'gpt2'
MISTRAL = 'mistralai/Mistral-7B-Instruct-v0.1'
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
            task = "question-answering"
            self.pipe = pipeline(task, model=self.model_name, torch_dtype=torch.bfloat16, device_map="auto")

    # Generate some text given a prompt
    def generate(self, prompt):
        # Use pipeline if using Tiny LLAMA
        if(self.model_name == TINY_LLAMA):
            template = [{"role": "user", "content": prompt}]

            # Don't tokenize output
            tokenized = self.pipe.tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)
            output = self.pipe(tokenized, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            return output[0]["generated_text"]

        # Otherwise use standard functions for LLMs
        else:
            input_ids_pristine = self.tokenizer.encode(prompt, return_tensors='pt')
            output_pristine_ids = self.model.generate(
                input_ids_pristine,
                max_length=150,  # or any desired max length
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id 
            )
            output_pristine = self.tokenizer.decode(output_pristine_ids[0], skip_special_tokens=True)
            return output_pristine
    
    # Generate a pluralistic answer given contextual passages from each belief group
    def generate_pluralistic(self, situation, passages, beliefgroups):
        number_passages = len(passages)
        prompt = "Comment on the following situation given the " + str(number_passages) + " possible choices. Your response should take into account all perspectives: " + ', '.join(beliefgroups) + "\n"
        prompt += "SITUATION: " + situation + "\n"
        for i, passage in enumerate(passages):
            prompt += "PASSAGE " + str(i + 1) + ": "+ passage + "\n"
        return self.generate(prompt)
    
    # Generate a pluralistic answer without contextual passages from each belief group
    def generate_dummy_pluralistic(self, situation):
        prompt = "Comment on the following situation. Consider diverse perspectives and provide a balanced, pluralistic response that respects different viewpoints.\n"
        prompt += "SITUATION: " + situation + "\n"
        return self.generate(prompt)
    
    # Generate an answer, without further instructions
    def generate_vanilla(self, situation):
        prompt = "Comment on the following situation"
        prompt += "SITUATION: " + situation + "\n"
        return self.generate(prompt)

    # Run entire pipeline:
    # - given data and kb
    # - retrieve closest passages in each belief group
    # - Fetch 1) pluralistic answer with passages 2) pluralistic answer without passages 3) answer with additional instructions
    def run_on_ds(self, ds, kb_embed, model_embd, distance_metric, beliefgroups, range = 2):
        results = []

        for i, elem in (enumerate(ds[:range])):
            query = elem[0]['situation'] + ' ' + elem[1]['intention']
            retrieved_moral = kb_embed.retrieve(model_embd.encode(query), distance_metric, 'moral', k = 1)
            retrieved_immoral = kb_embed.retrieve(model_embd.encode(query), distance_metric, 'immoral', k = 1)

            moral_choice_pred = ds[retrieved_moral[0]][0]['moral_action']
            immoral_choice_pred = ds[retrieved_immoral[0]][1]['immoral_action']

            pluralistic = self.generate_pluralistic(query, [moral_choice_pred, immoral_choice_pred], beliefgroups)
            dummy_pluralistic = self.generate_dummy_pluralistic(query)
            vanilla = self.generate_vanilla(query)

            results.append({"pluralistic": pluralistic, "dummy_pluralistic": dummy_pluralistic, "vanilla": vanilla})

        return results

        
    