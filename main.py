from AnswerWProba import *
from opinionQA import *

if __name__ == "__main__":
    # Specify paths
    dataset_path = "opinion_qa.csv"  # Update this to your Opinion QA dataset file
    output_path = "opinion_qa_results.csv"  # Output file for results
    
    # Initialize the framework with your chosen model
    model_name = "gpt2"  # Replace with your desired model
    aligner = AnswerWProba(model_name)
    
    # Process the Opinion QA dataset
    process_opinion_qa(aligner, dataset_path, output_path)