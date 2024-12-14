import pandas as pd

# Dataset Iteration Functionality
def load_opinion_qa_dataset(filepath):
    """
    Load the Opinion QA dataset from a file.
    """
    return pd.read_csv(filepath)

def process_opinion_qa(aligner, dataset_path, output_path, num_return_sequences=5, top_p=0.9):
    """
    Iterate over the Opinion QA dataset, generate answers, and save results to a file.
    """
    dataset = load_opinion_qa_dataset(dataset_path)
    
    if 'question' not in dataset.columns:
        raise ValueError("Dataset must contain a 'question' column.")
    
    results = []
    for index, row in dataset.iterrows():
        question = row['question']
        print(f"Processing question {index+1}/{len(dataset)}: {question}")
        
        answers = aligner.get_multiple_answers(
            prompt=question, 
            num_return_sequences=num_return_sequences, 
            top_p=top_p
        )
        
        for i, (answer, prob) in enumerate(answers):
            results.append({
                "question": question,
                "answer_number": i + 1,
                "answer": answer,
                "log_probability": prob
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index = False)
    print(f"Results saved to {output_path}")
