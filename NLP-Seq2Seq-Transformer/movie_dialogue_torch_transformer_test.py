
import time
import numpy as np
import pandas as pd
import pickle as pkl

import torch
import torch_transformer_network as transformer_model

# Define the weight update step for multiple sub-batches. #
def sub_batch_train_step(
    model, sub_batch_sz, 
    x_encode, x_decode, x_output, 
    optimizer, grad_clip=1.0, p_keep=0.90):
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    
    batch_size = x_encode.shape[0]
    if batch_size <= sub_batch_sz:
        sub_batch = 1
    else:
        sub_batch = int(batch_size / sub_batch_sz) + 1
    avg_batch_loss = 0.0
    
    optimizer.zero_grad()
    for n_sub in range(sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_encode = torch.tensor(
            x_encode[id_st:id_en, :], dtype=torch.long)
        tmp_decode = torch.tensor(
            x_decode[id_st:id_en, :], dtype=torch.long)
        tmp_output = torch.tensor(
            x_output[id_st:id_en, :], dtype=torch.long)
        
        if torch.cuda.is_available():
            tmp_encode = tmp_encode.cuda()
            tmp_decode = tmp_decode.cuda()
            tmp_output = tmp_output.cuda()
        
        output_logits = model(
            tmp_encode, tmp_decode, p_keep=p_keep)
        tmp_losses = torch.sum(torch.sum(ce_loss_fn(
            torch.transpose(output_logits, 1, 2), tmp_output), 1))
        sub_batch_loss = tmp_losses / batch_size
        
        # Accumulate the gradients. #
        sub_batch_loss.backward()
        avg_batch_loss += sub_batch_loss.item()
    
    # Update using the optimizer. #
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), grad_clip)
    optimizer.step()
    return avg_batch_loss

# Model Parameters. #
batch_size = 256
sub_batch  = 64
seq_encode = 10
seq_decode = 11

num_layers  = 3
num_heads   = 4
prob_keep   = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size

tmp_path = "C:/Users/admin/Desktop/Codes/"
model_ckpt_dir  = \
    "C:/Users/admin/Desktop/PyTorch_Models/transformer_seq2seq"

tmp_pkl_file = tmp_path + "movie_dialogues.pkl"
with open(tmp_pkl_file, "rb") as tmp_file_load:
    data_tuple = pkl.load(tmp_file_load)
    idx2word = pkl.load(tmp_file_load)
    word2idx = pkl.load(tmp_file_load)
vocab_size = len(word2idx)

num_data  = len(data_tuple)
SOS_token = word2idx["SOS"]
EOS_token = word2idx["EOS"]
PAD_token = word2idx["PAD"]
UNK_token = word2idx["UNK"]

# Set the number of threads. #
torch.set_num_threads(1)

print("Building the Transformer Model.")
start_time = time.time()

seq2seq_model = transformer_model.TransformerNetwork(
    num_layers, num_heads, hidden_size, ffwd_size, 
    vocab_size, vocab_size, seq_encode, seq_decode, 
    embed_size=hidden_size, attn_type="mult_attn")
if torch.cuda.is_available():
    seq2seq_model.cuda()

elapsed_time = (time.time() - start_time) / 60
print("Transformer Model built (" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = torch.load(model_ckpt_dir)
n_iter = ckpt["step"]
seq2seq_model.load_state_dict(ckpt['model_state_dict'])

# Placeholders to store the batch data. #
tmp_test_in = np.zeros([1, seq_encode], dtype=np.int32)
tmp_test_dec = \
    SOS_token * np.ones([1, seq_decode], dtype=np.int32)

print("-" * 50)
print("Testing the Transformer Network", 
      "(" + str(n_iter), "iterations).")
print("Vocabulary Size:", str(vocab_size))

while True:
    tmp_phrase = input("Enter input phrase: ")
    if tmp_phrase == "":
        break
    else:
        tmp_phrase = tmp_phrase.lower()
        
        tmp_test_in[:, :] = PAD_token
        tmp_i_tok = tmp_phrase.split(" ")
        tmp_i_idx = [word2idx.get(x, UNK_token) for x in tmp_i_tok]
        
        n_input = len(tmp_i_idx)
        tmp_test_in[0, :n_input] = tmp_i_idx
        
        infer_in  = torch.tensor(tmp_test_in, dtype=torch.long)
        infer_dec = torch.tensor(tmp_test_dec, dtype=torch.long)
        if torch.cuda.is_available():
            infer_in  = infer_in.cuda()
            infer_dec = infer_dec.cuda()
        
        gen_ids = seq2seq_model(infer_in, infer_dec, infer=True)
        if torch.cuda.is_available():
            gen_ids = gen_ids.detach().cpu()
        gen_phrase = [idx2word[x] for x in gen_ids.numpy()[0]]
        gen_phrase = " ".join(gen_phrase)
        
        print("")
        print("Input Phrase:")
        print(tmp_phrase)
        print("Generated Phrase:")
        print(gen_phrase)
        print("")
