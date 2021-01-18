
import time
import numpy as np
import pandas as pd
import pickle as pkl

import torch
import torch_gpt_network as gpt_module

# Model Parameters. #
batch_size = 256
sub_batch  = 64
seq_length = 21
num_heads  = 4
num_layers = 3

prob_keep = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size
warmup_flag = True
cooling_step = 2000

tmp_path = "C:/Users/admin/Desktop/Codes/"
model_ckpt_dir  = \
    "C:/Users/admin/Desktop/PyTorch_Models/dialogue_transformer_gpt"

# Load the data. #
tmp_pkl_file = \
    "C:/Users/admin/Desktop/Codes/movie_dialogues.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    data_tuple = pkl.load(tmp_load_file)
    idx2word = pkl.load(tmp_load_file)
    word2idx = pkl.load(tmp_load_file)

vocab_size = len(word2idx)
print("Vocabulary Size:", str(vocab_size)+".")

tmp_data = []
for tmp_row in data_tuple:
    tmp_q = tmp_row[0].split(" ")
    tmp_a = tmp_row[1].split(" ")
    
    if (len(tmp_q) + len(tmp_a)) > 1 and \
        (len(tmp_q) + len(tmp_a)) <= seq_length:
        tmp_data.append(tmp_row)

num_data  = len(tmp_data)
SOS_token = word2idx["SOS"]
EOS_token = word2idx["EOS"]
PAD_token = word2idx["PAD"]
UNK_token = word2idx["UNK"]
print("Total of", str(len(tmp_data)), "rows loaded.")

# Set the number of threads to use. #
torch.set_num_threads(1)

# Build the Transformer. #
print("Building the GPT Model.")
start_time = time.time()

gpt_model = gpt_module.GPT_Network(
    num_layers, num_heads, 
    hidden_size, ffwd_size, vocab_size, 
    seq_length, embed_size=hidden_size, 
    p_keep=prob_keep, attn_type="mult_attn")
if torch.cuda.is_available():
    gpt_model.cuda()

elapsed_time = (time.time()-start_time) / 60
print("GPT Model Built", "("+str(elapsed_time)+" mins).")

ckpt = torch.load(model_ckpt_dir)
n_iter = ckpt["step"]

gpt_model.load_state_dict(ckpt['model_state_dict'])

# Train the Transformer model. #
tmp_test_in = np.zeros([1, seq_length], dtype=np.int32)

print("-" * 50)
print("Training the GPT Network", 
      "(" + str(n_iter) + " iterations).")
print(str(num_data), "training samples.")
print("-" * 50)

# Update the neural network's weights. #
while True:
    tmp_phrase = input("Enter input phrase: ")
    if tmp_phrase == "":
        break
    else:
        tmp_phrase = tmp_phrase.lower()
        
        tmp_test_in[:, :] = PAD_token
        tmp_p_tokens = tmp_phrase.split(" ")
        
        tmp_p_idx = [word2idx.get(
            x, UNK_token) for x in tmp_p_tokens]
        n_tokens  = len(tmp_p_idx) + 1
        tmp_test_in[0, :n_tokens] = tmp_p_idx + [SOS_token]
        
        infer_in = torch.tensor(
            tmp_test_in[:, :n_tokens], dtype=torch.long)
        if torch.cuda.is_available():
            infer_in = infer_in.cuda()
        
        tmp_infer = gpt_module.infer(
            gpt_model, infer_in, seq_length)
        if torch.cuda.is_available():
            tmp_infer = tmp_infer.detach().cpu()
        
        gen_phrase = [
            idx2word[x] for x in tmp_infer[0].numpy()]
        gen_output = " ".join(gen_phrase[n_tokens:])
        gen_phrase = " ".join(gen_phrase)
        
        print("")
        print("Input Phrase:")
        print(" ".join(tmp_p_tokens))
        print("Generated Phrase:")
        print(gen_phrase)
        print("Generated Response:")
        print(gen_output)
        print("")
        del n_tokens

