import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import pipeline
from rag import *

# Global variables for models
GPT2 = 'gpt2'
TINY_LLAMA = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
GEMMA = 'google/gemma-2-2b'

class llm():
    def __init__(self, model_name = GPT2, task = "text-generation"):
        super().__init__()
        self.model_name = model_name
        self.task = task

        # Initialize LLM
        if(model_name == GPT2):
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        if(model_name == TINY_LLAMA):
            self.pipe = pipeline(task, model=self.model_name, torch_dtype=torch.bfloat16, device_map="auto")
        if(model_name == GEMMA):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)  

    # Generate some text given a prompt
    def generate(self, prompt):
        # We choose to use pipeline module here

        # ============================= TINY LLAMA ================================
        if((self.model_name == TINY_LLAMA) and self.task == "text-generation"):
            template = [{"role": "user", "content": prompt}]
            tokenized = self.pipe.tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)
            output = self.pipe(tokenized, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            return output[0]["generated_text"]
        
        
        # ============================= OTHER MODELS ================================
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
        prompt = "Comment on the following situation given the " + str(number_passages) + " possible choices. Your response should take into account all perspectives: " + ', '.join(beliefgroups) + ".\n"
        prompt += "SITUATION: " + situation + "\n"
        for i, passage in enumerate(passages):
            prompt += "PASSAGE " + str(i + 1) + ": "+ passage + "\n"
        return self.generate(prompt), len(prompt)
    
    # Generate a pluralistic answer without contextual passages from each belief group
    def generate_dummy_pluralistic(self, situation):
        prompt = "Comment on the following situation. Consider diverse perspectives and provide a balanced, pluralistic response that respects different viewpoints.\n"
        prompt += "SITUATION: " + situation + "\n"
        return self.generate(prompt), len(prompt)
    
    # Generate an answer, without further instructions
    def generate_vanilla(self, situation):
        prompt = "Comment on the following situation.\n"
        prompt += "SITUATION: " + situation + "\n"
        return self.generate(prompt), len(prompt)

    def generate_single_belief(self, situation, passage, beliefgroup):
        prompt = "Comment on the following situation given the accompanying passage. Your response should reflect the " + beliefgroup + " viewpoint in the passage. \n"
        prompt += "SITUATION: " + situation + " viewpoint. \n"
        prompt += "PASSAGE : " + passage + "\n"
        return self.generate(prompt), len(prompt)
        

    # Run entire pipeline:
    # - given data and kb
    # - retrieve closest passages in each belief group
    # - Fetch 1) pluralistic answer with passages 2) pluralistic answer without passages 3) answer with additional instructions
    def run_on_ds(self, ds, kb_embed, model_embd, distance_metric, beliefgroups, range = 2, trim = True):
        results = []
        for i, elem in (enumerate(ds[:range])):
            query = elem[0]['situation'] + ' ' + elem[1]['intention']
            retrieved_moral = kb_embed.retrieve(model_embd.encode(query), distance_metric, 'moral', k = 1)
            retrieved_immoral = kb_embed.retrieve(model_embd.encode(query), distance_metric, 'immoral', k = 1)

            moral_choice_pred = ds[retrieved_moral[0]][0]['moral_action']
            immoral_choice_pred = ds[retrieved_immoral[0]][1]['immoral_action']
            
            pluralistic, length_prompt_plur = self.generate_pluralistic(query, [moral_choice_pred, immoral_choice_pred], beliefgroups)
            dummy_pluralistic, length_prompt_dummy_plur = self.generate_dummy_pluralistic(query)
            vanilla, length_prompt_van = self.generate_vanilla(query)
            moral, length_prompt_moral = self.generate_single_belief(query, moral_choice_pred, beliefgroups[0])
            immoral, length_prompt_immoral = self.generate_single_belief(query, immoral_choice_pred, beliefgroups[1])         
            
            if(trim):
                if (self.model_name == GPT2 or self.model_name == GEMMA):
                    ignore_length_plur = length_prompt_plur
                    ignore_length_dummy_plur = length_prompt_dummy_plur
                    ignore_length_van = length_prompt_van
                    ignore_length_moral = length_prompt_moral
                    gnore_length_immoral = length_prompt_immoral
                if (self.model_name == TINY_LLAMA):
                    ignore_length_plur = length_prompt_plur + len("<|user|>/n</s><|assistant|>")
                    ignore_length_dummy_plur = length_prompt_dummy_plur + len("<|user|>/n</s><|assistant|>")
                    ignore_length_van = length_prompt_van + len("<|user|>/n</s><|assistant|>")
                    ignore_length_moral = length_prompt_moral + len("<|user|>/n</s><|assistant|>")
                    gnore_length_immoral = length_prompt_immoral + len("<|user|>/n</s><|assistant|>")
                
                pluralistic = pluralistic[ignore_length_plur:].strip()
                dummy_pluralistic = dummy_pluralistic[ignore_length_dummy_plur:].strip()
                vanilla = vanilla[ignore_length_van:].strip()
                moral = moral[ignore_length_moral:].strip()
                immoral = immoral[gnore_length_immoral:].strip()

            results.append({"pluralistic": pluralistic, "dummy_pluralistic": dummy_pluralistic, "vanilla": vanilla, "moral": moral, "immoral": immoral})

        return results

