
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

initial_lr    = 0.001
gradient_clip = 1.00
maximum_iter  = 10000
restore_flag  = False
display_step  = 100
cooling_step  = 1000
warmup_steps  = 500
anneal_step   = 2000
anneal_rate   = 0.75

tmp_path = "C:/Users/admin/Desktop/Codes/"
model_ckpt_dir  = \
    "C:/Users/admin/Desktop/PyTorch_Models/transformer_seq2seq"
train_loss_file = "C:/Users/admin/Desktop/Codes/"
train_loss_file += "train_loss_dialogue_transformer_torch.csv"

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

# We change the optimizer to AdamW as Adam performs poorly #
# when training the model.
#seq2seq_optimizer = torch.optim.Adam(
#    seq2seq_model.parameters(), 
#    betas=(0.9, 0.98), eps=1.0e-9)
seq2seq_optimizer = \
    torch.optim.AdamW(seq2seq_model.parameters())

elapsed_time = (time.time() - start_time) / 60
print("Transformer Model built (" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
if restore_flag:
    ckpt = torch.load(model_ckpt_dir)
    n_iter = ckpt["step"]
    
    seq2seq_model.load_state_dict(ckpt['model_state_dict'])
    seq2seq_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    
    train_loss_df = pd.read_csv(train_loss_file)
    train_loss_list = [tuple(
        train_loss_df.iloc[x].values) \
            for x in range(len(train_loss_df))]
else:
    print("Training a new model.")
    n_iter = 0
    train_loss_list = []

# Placeholders to store the batch data. #
tmp_input   = np.zeros([batch_size, seq_encode], dtype=np.int32)
tmp_seq_out = np.zeros([batch_size, seq_decode+1], dtype=np.int32)
tmp_test_in = np.zeros([1, seq_encode], dtype=np.int32)
tmp_test_dec = SOS_token * np.ones([1, seq_decode], dtype=np.int32)

print("-" * 50)
print("Training the Transformer Network", 
      "(" + str(n_iter), "iterations).")
print("Vocabulary Size:", str(vocab_size))
print("No. of data:", str(len(data_tuple)))

tot_loss = 0.0
start_tm = time.time()
while n_iter < maximum_iter:
    step_val = float(max(n_iter+1, warmup_steps))**(-0.5)
    learn_rate_val = float(hidden_size)**(-0.5) * step_val
    
    for tmp_opt_group in \
        seq2seq_optimizer.param_groups:
        tmp_opt_group["lr"] = learn_rate_val
    
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    tmp_input[:, :]   = PAD_token
    tmp_seq_out[:, :] = PAD_token
    tmp_seq_out[:, 0] = SOS_token
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_i_tok = data_tuple[tmp_index][0].split(" ")
        tmp_o_tok = data_tuple[tmp_index][1].split(" ")

        tmp_i_idx = [word2idx.get(x, UNK_token) for x in tmp_i_tok]
        tmp_o_idx = [word2idx.get(x, UNK_token) for x in tmp_o_tok]

        n_input  = len(tmp_i_idx)
        n_output = len(tmp_o_idx)
        n_decode = n_output + 1

        tmp_input[n_index, :n_input] = tmp_i_idx
        tmp_seq_out[n_index, 1:n_decode] = tmp_o_idx
        tmp_seq_out[n_index, n_decode] = EOS_token

    tmp_decode = tmp_seq_out[:, :-1]
    tmp_output = tmp_seq_out[:, 1:]
    
    tmp_loss = sub_batch_train_step(
        seq2seq_model, sub_batch, 
        tmp_input, tmp_decode, tmp_output, 
        seq2seq_optimizer, p_keep=prob_keep)
    
    n_iter += 1
    tot_loss += tmp_loss

    if n_iter % display_step == 0:
        end_time = time.time()
        avg_loss = tot_loss / display_step
        tot_loss = 0.0
        elapsed_tm = (end_time - start_tm) / 60
        start_tm   = time.time()

        tmp_test_in[:, :] = PAD_token
        sample_id = np.random.choice(num_data, size=1)
        tmp_data  = data_tuple[sample_id[0]]

        tmp_i_tok = tmp_data[0].split(" ")
        tmp_o_tok = tmp_data[1].split(" ")
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

        print("Iteration", str(n_iter) + ":")
        print("Elapsed Time:", str(elapsed_tm) + " mins.")
        print("Average Loss:", str(avg_loss))
        print("Gradient Clip:", str(gradient_clip))
        print("Learning Rate:", str(tmp_opt_group["lr"]))
        
        print("")
        print("Input Phrase:")
        print(tmp_data[0])
        print("Generated Phrase:")
        print(gen_phrase)
        print("Actual Response:")
        print(tmp_data[1])
        print("")
        
        # Save the training progress. #
        train_loss_list.append((
            n_iter, avg_loss, 
            tmp_data[0], gen_phrase, tmp_data[1]))
        train_cols_df = [
            "n_iter", "xent_loss", 
            "input_phrase", "gen_phrase", "out_phrase"]
        train_loss_df = pd.DataFrame(
            train_loss_list, columns=train_cols_df)
        train_loss_df.to_csv(train_loss_file, index=False)
        
        # Save the model. #
        torch.save({
            'step': n_iter,
            'model_state_dict': seq2seq_model.state_dict(),
            'optimizer_state_dict': seq2seq_optimizer.state_dict()
            }, model_ckpt_dir)
        print("Saved model to:", model_ckpt_dir)
        print("-" * 50)
    
    if n_iter % cooling_step == 0:
        print("Cooling the CPU for 2 minutes.")
        time.sleep(120)
        print("Resuming training.")


