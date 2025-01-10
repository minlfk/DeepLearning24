#Norm : in the dataset compare it to the pluralistic answers of 
import math
import numpy as np
from typing import Literal, List, Any
from enum import Enum
from sentence_transformers import SentenceTransformer, util, SimilarityFunction
from rag import *

def return_distance(model_embd, text1, text2, similarity_metric):
    if(similarity_metric == "cosine"):
        model_embd.similarity_fn_name = SimilarityFunction.COSINE
    elif similarity_metric == "dot":
        model_embd.similarity_fn_name = SimilarityFunction.DOT_PRODUCT    
    elif similarity_metric == "euclidean":
        model_embd.similarity_fn_name = SimilarityFunction.EUCLIDEAN   
    elif similarity_metric == "manhattan":
        model_embd.similarity_fn_name = SimilarityFunction.MANHATTAN 

    embd1 = model_embd.encode(text1)
    embd2 = model_embd.encode(text2)

    similarity = model_embd.similarity([embd1], [embd2])
    return similarity[0][0].item()


# Measure the distance between some responses by LLM and the average of actual passages from different belief groups
def similarities_results_llm(model_embd, results, key, passages, similarity_metric):
    
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

        # Compute embedding of LLM output
        llm_embd = model_embd.encode(result[key])
        # Compute average of embeddings of passages from different perspectives
        beliefs_embds = [model_embd.encode[passage] for passage in passages[i]]

        average_embd = np.mean(beliefs_embds, axis=0)

        # Compute similarity 
        similarity = model_embd.similarity([llm_embd], [average_embd])
        similarity_scores.append(similarity[0][0].item())  # Add the similarity score to the list

    return similarity_scores

# Measure distance between LLM output and norm it needs to respect
def similarities_norm(model_embd, results, key, norms, similarity_metric="cosine"):
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

    for i, result in enumerate(results):
        
        # Compute embeddings of LLM output and the norm to respect
        llm_embd = model_embd.encode(result[key])
        norm_embd = model_embd.encode(norms[i])

        # Compute similarity 
        similarity = model_embd.similarity([llm_embd], [norm_embd])
        similarity_scores.append(similarity[0][0].item())  # Add the similarity score to the list

    return similarity_scores