import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, TopPLogitsWarper

# Core Framework
class AnswerWProba:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_multiple_answers(self, prompt, max_length=50, num_return_sequences=5, top_p=0.9):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        logits_processor = LogitsProcessorList([TopPLogitsWarper(top_p=top_p)])
        
        # Generate multiple answers
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            logits_processor=logits_processor
        )
        
        # Decode answers and calculate probabilities
        answers = []
        for output in outputs:
            decoded_answer = self.tokenizer.decode(output, skip_special_tokens=True)
            log_probs = self.calculate_probability(output, inputs)
            answers.append((decoded_answer, log_probs))
        
        return answers

    def calculate_probability(self, output, inputs):
        # Calculate token-level log probabilities
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        token_probs = probabilities[0, torch.arange(len(output)), output].cpu().numpy()
        total_log_prob = sum([torch.log(torch.tensor(prob)).item() for prob in token_probs])
        return total_log_prob