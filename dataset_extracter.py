""" dataset_extracter.py contains the functions necessary for creating the dataset for our
    experiments. 
    
    Note, much of our code related to the GPT-2 and BERT models comes from the following
    github repo: https://github.com/huggingface/pytorch-pretrained-BERT
    
"""

import torch
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from tqdm import tqdm
import random as rand
import os
import glob
import pickle

# generate_text_inputs produces num_inputs number of text samples for passing through models
def generate_text_inputs(text_repo_path, save_to, num_inputs, input_len=None, mode="penn-treebank"):
    text_inputs = []
    
    if mode == "penn-treebank":
        print("in if statement ... ")
        for filename in glob.glob(os.path.join(text_repo_path, "wsj_*"))[:num_inputs]:
            print("inspecting filename == ", filename)
            with open(filename, 'r') as f:
                text = ""
                for line in f.readlines()[2:]:
                    text = text + line
                text_inputs.append(text)
    
    else:
        raise ValueError("Only penn-treebank mode is currently supported")
    
    with open(save_to, "wb") as outfile:
        pickle.dump(text_inputs, outfile)
        print("Len text == ", len(text_inputs))
    pass
    # return text_inputs

# compute token-level layer activations for input text file in the gpt2 model
def extract_gpt2_hidden_activations(text_path, save_activs_to):#, mode="full_model", focus_layers=[]):
    text_inputs = []
    with open(text_path, "rb") as infile:
        text_inputs = pickle.load(infile)

    # num_inputs = len(text_inputs)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # get the hidden activations
    layer_activs = []
    for text in text_inputs:
        indexed_tokens = tokenizer.encode(text)
        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
        model = GPT2Model.from_pretrained('gpt2')
        model.eval()
        model.to('cuda')
        with torch.no_grad():
            hidden, _ = model(tokens_tensor)
            layer_activs.append(hidden.cpu().numpy().squeeze())
        torch.cuda.empty_cache()

    # save layer dimensions
    with open(save_activs_to, "wb") as outfile:
        pickle.dump(layer_activs, outfile)
    pass

# compute and save word-level activation sums for the gpt2 model
def extract_gpt2_hidden_word_representations(word, save_activs_to):#, mode="full_model", focus_layers=[]):
    # text_inputs = []
    # with open(text_path, "rb") as infile:
    #     text_inputs = pickle.load(infile)

    # num_inputs = len(text_inputs)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # get the hidden activations
    word_vec = None
    indexed_tokens = tokenizer.encode(word)
    num_tokens = len(indexed_tokens)
    tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
    model = GPT2Model.from_pretrained('gpt2')
    model.eval()
    model.to('cuda')
    with torch.no_grad():
        hidden, _ = model(tokens_tensor)
        hidden_np = hidden.cpu().numpy().squeeze()
        seq_dim = hidden_np.shape.index(num_tokens)
        word_vec = np.sum(hidden_np, axis=seq_dim).squeeze
    torch.cuda.empty_cache()

    # save layer dimensions
    with open(save_activs_to, "wb") as outfile:
        pickle.dump(word_vec, outfile)
    pass


def generate_matching_random_matrices(num_inputs, layer_activs_path, save_to):
    layer_activs = []
    with open(layer_activs_path, "rb") as infile:
        layer_activs = pickle.load(infile)
    # generate random matrices
    random_matrices = []
    for activ in layer_activs:
        r = np.random.random(activ.shape)
        random_matrices.append(r)
    
    # save random matrices
    with open(save_to, "wb") as outfile:
        pickle.dump(random_matrices, outfile)
    pass


if __name__ == "__main__":
    pass
