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
def generate_text_inputs(text_repo_path, save_to, num_inputs, input_len=100, mode="penn-treebank"):
    text_inputs = []
    
    if mode == "penn-treebank": # if we want to use the penn_treebank dataset found at https://www.kaggle.com/nltkdata/penn-tree-bank#treebank.zip
        # loop through penn-treebank dataset num_inputs times
        for filename in glob.glob(os.path.join(text_repo_path, "wsj_*"))[:num_inputs]:
            
            with open(filename, 'r') as f:
                text = ""
                
                # read in the lines of each text sample
                for line in f.readlines()[2:]:
                    total_len = len(text) + len(line) # represents the length of each text sample
                    if total_len <= input_len:
                        text = text + line
                    else:
                        break
                # save the text sample to the list we will output to a file
                text_inputs.append(text)
    
    else:
        raise ValueError("Only penn-treebank mode is currently supported")
    
    # save the text for experiments
    with open(save_to, "wb") as outfile:
        pickle.dump(text_inputs, outfile)
        print("Len text == ", len(text_inputs))
    pass
    # return text_inputs


# compute token-level layer activations for input text file in the gpt2 model
def extract_gpt2_hidden_activations(text_path, save_activs_to):#, mode="full_model", focus_layers=[]):
    # read in text samples to pass through single layer of gpt2 model
    text_inputs = []
    with open(text_path, "rb") as infile:
        text_inputs = pickle.load(infile)

    # num_inputs = len(text_inputs)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # get the hidden activations - assumes a gpu is available
    layer_activs = []
    for text in text_inputs:
        # tokenize text
        indexed_tokens = tokenizer.encode(text)
        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
        # set up model
        model = GPT2Model.from_pretrained('gpt2')
        model.eval()
        model.to('cuda')

        # grab the hidden activations and save them to layer_actives
        with torch.no_grad():
            hidden, _ = model(tokens_tensor)
            layer_activs.append(hidden.cpu().numpy().squeeze())
        
        # clear gpu memory in preparation for next text sample
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

    # get the hidden activations for word -- assumes gpu is available
    word_vec = None # initialize word vector object 
    
    # tokenize word
    indexed_tokens = tokenizer.encode(word)
    num_tokens = len(indexed_tokens)
    tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
    # set up model
    model = GPT2Model.from_pretrained('gpt2')
    model.eval()
    model.to('cuda')

    # get word_vec
    with torch.no_grad():
        # get token-wise activations
        hidden, _ = model(tokens_tensor)
        hidden_np = hidden.cpu().numpy().squeeze()

        # identify hidden layer dimension that represents different tokens
        # seq_dim = hidden_np.shape.index(num_tokens)
        seq_dim = 0 # we know that the dimension corresponding to tokens is the 1st dimension, indexed by 0
        
        # sum the hidden layer element-wise along the token dimension to get word vector representation
        word_vec = np.sum(hidden_np, axis=seq_dim)#.squeeze()

    # clear gpu memory
    torch.cuda.empty_cache()

    # save word vector
    with open(save_activs_to, "wb") as outfile:
        pickle.dump(word_vec, outfile)
    pass


def generate_matching_random_matrices(layer_activs_path, save_to):
    # read in hidden layer activations
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
