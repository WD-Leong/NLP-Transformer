
import time
import math
import numpy as np
import pandas as pd
import pickle as pkl
from collections import Counter
import byte_pair_encoding as bpe

import torch
import torch_gpt_module as gpt_module

# Custom functions. #
def compute_bleu_score(
    reference_corpus, translated_corpus, max_order=4, smooth=False):
    
    def _get_n_grams(segment, max_order):
        n_gram_counts = Counter()
        for order in range(1, max_order+1):
            for i in range(0, len(segment)-order+1):
                ngram = tuple(segment[i:(i+order)])
                n_gram_counts[ngram] += 1
        return n_gram_counts
    
    matches_by_order = [0]*max_order
    possible_matches_by_order = [0]*max_order
    
    reference_length   = 0
    translation_length = 0
    
    for (references, translation) in \
        zip(reference_corpus, translated_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)
        
        merged_ref_ngram_counts = Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_n_grams(reference, max_order)
        translated_ngram_counts = _get_n_grams(translation, max_order)
        
        overlap = translated_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]
        for order in range(1, max_order+1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches
    
    precisions = [0]*max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = \
                (matches_by_order[i]+1.0) / possible_matches_by_order[i]
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = \
                    float(matches_by_order[i]) / possible_matches_by_order[i]
            else:
                precisions[i] = 0.0
        
    if min(precisions) > 0:
        p_log_sum = \
            sum((1.0/max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0.0
    
    tmp_ratio = float(translation_length) / reference_length
    if tmp_ratio > 1.0:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (1.0/tmp_ratio))
    
    bleu = geo_mean*bp
    return bleu

# Define the weight update step. #
def sub_batch_train_step(
    model, sub_batch_sz, x_encode, x_output, 
    optimizer, grad_clip=1.0, p_keep=0.90):
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    
    batch_size = x_encode.shape[0]
    if batch_size <= sub_batch_sz:
        sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        sub_batch = int(batch_size / sub_batch_sz)
    else:
        sub_batch = int(batch_size / sub_batch_sz) + 1
    avg_losses = 0.0
    
    optimizer.zero_grad()
    for n_sub in range(sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_encode = torch.tensor(
            x_encode[id_st:id_en, :], dtype=torch.long)
        tmp_output = torch.tensor(
            x_output[id_st:id_en, :], dtype=torch.long)
        
        if torch.cuda.is_available():
            tmp_encode = tmp_encode.cuda()
            tmp_output = tmp_output.cuda()
        
        output_logits = model(tmp_encode, p_keep=p_keep)
        tmp_losses = torch.sum(torch.sum(ce_loss_fn(
            torch.transpose(output_logits, 1, 2), tmp_output), 1))
        sub_batch_loss = tmp_losses / batch_size
        
        # Accumulate the gradients. #
        sub_batch_loss.backward()
        avg_losses += sub_batch_loss.item()
    
    # Update using the optimizer. #
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), grad_clip)
    optimizer.step()
    return avg_losses

# Model Parameters. #
batch_size = 256
sub_batch  = 64
seq_length = 31
num_heads  = 4
num_layers = 3

gradient_clip = 1.00
maximum_iter  = 20000
restore_flag  = True
save_step     = 500
warmup_steps  = 4000
display_step  = 100
anneal_step   = 2500
anneal_rate   = 0.75

prob_keep = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size
warmup_flag = True
cooling_step = 2000

model_ckpt_dir  = \
    "C:/Users/admin/Desktop/PyTorch_Models/reddit_subword_gpt"
train_loss_file = "C:/Users/admin/Desktop/Codes/"
train_loss_file += "train_loss_gpt_reddit_subword_torch.csv"

# Load the data. #
tmp_pkl_file = \
    "C:/Users/admin/Desktop/Codes/reddit_jokes_subword.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    full_data = pkl.load(tmp_load_file)
    idx2subword = pkl.load(tmp_load_file)
    subword2idx = pkl.load(tmp_load_file)

vocab_size = len(subword2idx)
print("Vocabulary Size:", str(vocab_size)+".")

tmp_data = []
for tmp_row in full_data:
    if len(tmp_row) > 1 and \
        len(tmp_row) <= seq_length:
        tmp_data.append(tmp_row)

num_data  = len(tmp_data)
SOS_token = subword2idx["<SOS>"]
EOS_token = subword2idx["<EOS>"]
PAD_token = subword2idx["<PAD>"]
UNK_token = subword2idx["<UNK>"]
print("Total of", str(len(tmp_data)), "rows loaded.")

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

gpt_optimizer = \
    torch.optim.AdamW(gpt_model.parameters())
elapsed_time  = (time.time()-start_time) / 60
print("GPT Model Built", "("+str(elapsed_time)+" mins).")

# Create the model checkpoint. #
if restore_flag:
    ckpt = torch.load(model_ckpt_dir)
    n_iter = ckpt["step"]
    
    gpt_model.load_state_dict(ckpt['model_state_dict'])
    gpt_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    
    train_loss_df = pd.read_csv(train_loss_file)
    train_loss_list = [tuple(
        train_loss_df.iloc[x].values) \
        for x in range(len(train_loss_df))]
else:
    print("Training a new model.")
    n_iter = 0
    train_loss_list = []

# Train the Transformer model. #
tmp_out_seq = np.zeros(
    [batch_size, seq_length+1], dtype=np.int32)
tmp_test_in = np.zeros([1, seq_length], dtype=np.int32)

# Warmup learning schedule. #
if warmup_flag:
    step_min = float(max(n_iter, warmup_steps))**(-0.5)
    learning_rate = float(hidden_size)**(-0.5) * step_min
else:
    initial_lr = 0.001
    learning_rate = max(
        anneal_rate**(n_iter // anneal_step)*initial_lr, 1.0e-5)

print("-" * 50)
print("Training the GPT Network", 
      "(" + str(n_iter) + " iterations).")
print(str(num_data), "training samples.")
print("-" * 50)

# Update the neural network's weights. #
tot_loss = 0.0
start_tm = time.time()
while n_iter < maximum_iter:
    if warmup_flag:
        step_min = float(max(n_iter, warmup_steps))**(-0.5)
        learning_rate = float(hidden_size)**(-0.5) * step_min
    else:
        if n_iter % anneal_step == 0:
            learning_rate = max(
                anneal_rate**(n_iter // anneal_step)*initial_lr, 1.0e-6)
    
    # Select a sample from the data. #
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    tmp_out_seq[:, :] = PAD_token
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_p_idx = tmp_data[tmp_index]
        
        n_input = len(tmp_p_idx)
        tmp_out_seq[n_index, :n_input] = tmp_p_idx
        tmp_out_seq[n_index, n_input]  = EOS_token
    
    # Set the training data. #
    tmp_input  = tmp_out_seq[:, :-1]
    tmp_output = tmp_out_seq[:, 1:]
    
    # Set the training data. #
    tmp_input  = tmp_out_seq[:, :-1]
    tmp_output = tmp_out_seq[:, 1:]
    
    tmp_loss = sub_batch_train_step(
        gpt_model, sub_batch, 
        tmp_input, tmp_output, 
        gpt_optimizer, p_keep=prob_keep)

    n_iter += 1
    tot_loss += tmp_loss
    if n_iter % display_step == 0:
        end_tm = time.time()
        
        avg_loss = tot_loss / display_step
        tot_loss = 0.0
        elapsed_tm = (end_tm - start_tm) / 60
        
        tmp_test_in[:, :] = PAD_token
        sample_test = np.random.choice(num_data, size=1)
        tmp_p_index = tmp_data[sample_test[0]]
        
        in_phrase = bpe.bp_decode(tmp_p_index, idx2subword)
        in_phrase = " ".join(
            in_phrase).replace("<", "").replace(">", "")
        n_tokens  = len(tmp_p_index)
        n_sample  = np.random.randint(1, high=n_tokens-1)
        tmp_test_in[0, :n_tokens] = tmp_p_index
        
        infer_in = torch.tensor(
            tmp_test_in[:, :n_sample], dtype=torch.long)
        if torch.cuda.is_available():
            infer_in = infer_in.cuda()
        
        tmp_infer = gpt_module.infer(
            gpt_model, infer_in, seq_length)
        if torch.cuda.is_available():
            tmp_infer = tmp_infer.detach().cpu()
        del sample_test, n_tokens
        
        gen_phrase = bpe.bp_decode(
            tmp_infer[0].numpy(), idx2subword)
        gen_phrase = " ".join(
            gen_phrase).replace("<", "").replace(">", "")
        
        test_phrase = bpe.bp_decode(
            tmp_p_index[:n_sample], idx2subword)
        test_phrase = " ".join(
            test_phrase).replace("<", "").replace(">", "")
        
        print("Iteration", str(n_iter)+".")
        print("Elapsed Time:", str(elapsed_tm), "mins.")
        print("Gradient Clip:", str(gradient_clip)+".")
        print("Learning Rate:", str(learning_rate)+".")
        print("Average Loss:", str(avg_loss)+".")
        
        print("")
        print("Input Phrase:")
        print(test_phrase)
        print("Generated Phrase:")
        print(gen_phrase)
        print("Actual Phrase:")
        print(in_phrase)
        del n_sample
        
        train_loss_list.append((n_iter, avg_loss))
        start_tm = time.time()
        print("-" * 50)
    
    # Save the model. #
    if n_iter % save_step == 0:
        # Save the model. #
        torch.save({
            'step': n_iter,
            'model_state_dict': gpt_model.state_dict(),
            'optimizer_state_dict': gpt_optimizer.state_dict()
            }, model_ckpt_dir)
        print("Saved model to:", model_ckpt_dir)
        
        tmp_df_losses = pd.DataFrame(
            train_loss_list, columns=["n_iter", "xent_loss"])
        tmp_df_losses.to_csv(train_loss_file, index=False)
        del tmp_df_losses
    
    # Cool the GPU. #
    if n_iter % cooling_step == 0:
        print("Cooling GPU for 3 minutes.")
        time.sleep(180)
        print("Resume Training.")
        print("-" * 50)

