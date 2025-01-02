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
        
    def encode(self, string):
        return self.tokenizer.encode(string, return_tensors='pt')

    def generate_pristine(self, prompt):
        input_ids_pristine = self.tokenizer.encode(prompt, return_tensors='pt')
        output_pristine_ids = self.model.generate(input_ids_pristine)
        output_pristine = self.tokenizer.decode(output_pristine_ids[0], skip_special_tokens=True)
        print(output_pristine)
        return output_pristine
    
    #def rag_qa_pipe(question: str, passages: List[str]) -> str:
    #"""
    #Define the RAG pipeline which concatenates passages to the question.
    #:param question: Question text.
    #:param passages: Relevant text passages.
    #:return: Generated text from the pipeline.
    #"""
    #passages = "\n".join([f"CONTEXT: {c}" for c in passages])
    #return vanilla_qa_pipe(f"{passages}\nQUESTION: {question}\nANSWER: ", max_new_tokens=10)[0]["generated_text"]