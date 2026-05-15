from utilities.prepare_data import evalPairs
from query_embedding_space import embeddingSpace
import numpy as np
import os

CSV = "data/Hikam_Aligned_Passages.csv"
COL_NAMES = ["HIKMA TEXT", "VERSIFICATION WITNESS 1 TEXT", "VARIANT MARGINALIA WITNESS 1"]
GRAPH_OUT = "graphs/hikam/"
EMBEDDING_MODELS = ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
                    "sentence-transformers/LaBSE",
                    {"model": "intfloat/multilingual-e5-large", "prefix": "query: "},
                    "Omartificial-Intelligence-Space/Arabic-mpnet-base-all-nli-triplet",
                    "Omartificial-Intelligence-Space/Arabic-labse-Matryoshka"
                    ]               
CACHE_PATH = "data/cached_embeds.npz"
EVAL_PATH = "evaluations_hikam/"
MODEL_SCORES= "evaluations/hikam_model_scores.csv"
PASSIM_DATA_PATH = "data/passim_hikam"

def run_eval_sequence():
    
    # Set up the eval_pairs
    eval_pairs = evalPairs(CSV)
    eval_pairs.create_eval_groups(COL_NAMES)


    # Pass the eval set to the embedding space
    embedding_space = embeddingSpace(eval_pairs, EMBEDDING_MODELS, cache_path=CACHE_PATH)

    # # Change the dimensionality reduction
    # embedding_space.set_d_reduction("pacmap")
    


    # Run the full evaluation - just focus on passes and those that pass and fail some models
    embedding_space.run_full_evaluation(eval_pairs.group_labels, EVAL_PATH, n_fail=0, top_models=2, bottom_models=1)

def build_passim():
    if not os.path.exists(PASSIM_DATA_PATH):
        os.mkdir(PASSIM_DATA_PATH)
    

    eval_pairs = evalPairs(CSV)
    eval_pairs.create_concat_texts(COL_NAMES, PASSIM_DATA_PATH)
    



if __name__ == "__main__":
    run_eval_sequence()
    # build_passim()