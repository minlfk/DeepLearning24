#Norm : in the dataset compare it to the pluralistic answers of 
import math
import numpy as np
from typing import Literal, List, Any
from enum import Enum
from sentence_transformers import SentenceTransformer, util, SimilarityFunction
from rag import *


# Measure the distance between the pluralistic response and the average of purely moral and immoral responses from the llm
def similarities_results_llm(model_embd, results, pluralistic_key, norm_key, similarity_metric):
    """
    Calculate the embedding similarities between pluralistic and moral sentences for all results.
    :param model_embd: Pretrained Sentence Transformer model.
    :param results: List of results containing pluralistic and moral sentences.
    :param pluralistic_key: The key for pluralistic sentences in the results.
    :param norm_key: The key for moral sentences in the results.
    :similarity_metric: The similarity metric to use ("cosine", "dot", "euclidean", "manhattan"). By default, "cosine"
    :return: List of similarity scores.
    """
    
    similarity_scores = []
    
    if(similarity_metric == "cosine"):
        model_embd.similarity_fn_name = SimilarityFunction.COSINE
    elif similarity_metric == "dot":
        model_embd.similarity_fn_name = SimilarityFunction.DOT_PRODUCT    
    elif similarity_metric == "euclidean":
        model_embd.similarity_fn_name = SimilarityFunction.EUCLIDEAN   
    elif similarity_metric == "manhattan":
        model_embd.similarity_fn_name = SimilarityFunction.MANHATTAN 
    
    for i, result in enumerate(results):

        # Extract sentences
        pluralistic_sentence = result[pluralistic_key]
        norm_sentence = result[norm_key]

        # Compute embeddings for both sentences
        pluralistic_embd = model_embd.encode(pluralistic_sentence)
        norm_embd = model_embd.encode(norm_sentence)

        # Compute similarity 
        similarity = model_embd.similarity([pluralistic_embd], [norm_embd])
        similarity_scores.append(similarity[0][0].item())  # Add the similarity score to the list

    return similarity_scores

# Mesurer la distance entre réponses pluralistic et “norme” 
def similarities_norm(model_embd, results, pluralistic_key, norm_list, similarity_metric="cosine"):
    """
    Calculate the embedding similarities between pluralistic responses and norm sentences.
    :param model_embd: Pretrained Sentence Transformer model.
    :param results: List of results containing pluralistic responses.
    :param pluralistic_key: The key for pluralistic responses in the results.
    :param norm_list: The list of norm sentences.
    :param similarity_metric: The similarity metric to use ("cosine", "dot", etc.).
    :return: List of similarity scores (one for each result).
    """
    similarity_scores = []

    # Set the similarity function
    if similarity_metric == "cosine":
        model_embd.similarity_fn_name = SimilarityFunction.COSINE
    elif similarity_metric == "dot":
        model_embd.similarity_fn_name = SimilarityFunction.DOT_PRODUCT
    elif similarity_metric == "euclidean":
        model_embd.similarity_fn_name = SimilarityFunction.EUCLIDEAN
    elif similarity_metric == "manhattan":
        model_embd.similarity_fn_name = SimilarityFunction.MANHATTAN

    for result in results:
        # Extract the pluralistic sentence and encode it
        pluralistic_sentence = result[pluralistic_key]
        pluralistic_embd = model_embd.encode(pluralistic_sentence)

        # Compute similarities with all norm sentences and take the average
        norm_similarities = []
        for norm_sentence in norm_list:
            norm_embd = model_embd.encode(norm_sentence)
            similarity = model_embd.similarity([pluralistic_embd], [norm_embd])
            norm_similarities.append(similarity[0][0].item())  # Extract the scalar similarity value

        # Average the similarities with norm sentences
        avg_similarity = sum(norm_similarities) / len(norm_similarities)
        similarity_scores.append(avg_similarity)

    return similarity_scores