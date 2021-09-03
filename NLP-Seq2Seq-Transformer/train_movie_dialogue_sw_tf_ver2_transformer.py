import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tf_ver2_transformer_keras as tf_transformer

# Define the subword decoder. #
def bp_decode(indices_in, idx2subword):
    sw_idx_list = [idx2subword[x] for x in indices_in]
    words_list  = []
    
    curr_sw = ""
    for n_sw in range(len(sw_idx_list)):
        tmp_sw = sw_idx_list[n_sw]
        if tmp_sw.find("<") != -1 \
            and tmp_sw.find(">") != -1:
            tmp_word = tmp_sw
            curr_sw = ""
            words_list.append(tmp_word)
        elif tmp_sw.find(">") != -1 \
            and tmp_sw.find("<") == -1:
            curr_sw += tmp_sw
            tmp_word = curr_sw
            curr_sw = ""
            words_list.append(tmp_word)
        elif tmp_sw.find(">") == -1 \
            and tmp_sw.find("<") != -1:
            curr_sw += tmp_sw
        else:
            curr_sw += tmp_sw
    return words_list

# Define the weight update step. #
def train_step(
    model, sub_batch_sz, 
    x_encode, x_decode, x_output, optimizer, 
    learning_rate=1.0e-3, grad_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_encode.shape[0]
    if batch_size <= sub_batch_sz:
        n_sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        n_sub_batch = int(batch_size / sub_batch_sz)
    else:
        n_sub_batch = int(batch_size / sub_batch_sz) + 1
    
    model_params  = model.trainable_variables
    acc_gradients = [
        tf.zeros_like(var) for var in model_params]
    
    tot_losses = 0.0
    for n_sub in range(n_sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (n_sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_encode = x_encode[id_st:id_en, :]
        tmp_decode = x_decode[id_st:id_en, :]
        tmp_output = x_output[id_st:id_en, :]
        with tf.GradientTape() as grad_tape:
            output_logits = model(tmp_encode, tmp_decode)
            
            tmp_losses = tf.reduce_sum(tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_output, logits=output_logits), axis=1))
        
        tot_losses += tmp_losses
        tmp_gradients = grad_tape.gradient(
            tmp_losses, model_params)
        acc_gradients = [tf.add(
            acc_grad, grad) for acc_grad, grad \
                in zip(acc_gradients, tmp_gradients)]
    
    avg_losses = tot_losses / batch_size
    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    optimizer.apply_gradients(zip(acc_gradients, model_params))
    return avg_losses

# Model Parameters. #
sub_batch_sz = 64
batch_size = 256
seq_encode = 25
seq_decode = 26

num_layers  = 3
num_heads   = 4
prob_keep   = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size

initial_lr    = 0.001
gradient_clip = 1.00
maximum_iter  = 50000
restore_flag  = True
display_step  = 100
cooling_step  = 500
warmup_steps  = 5000
anneal_step   = 2000
anneal_rate   = 0.75

tmp_path = "C:/Users/admin/Desktop/"
model_ckpt_dir  = "../TF_Models/transformer_sw_keras"
train_loss_file = "train_loss_sw_transformer_keras.csv"

tmp_pkl_file = "movie_dialogues_subword_gpt.pkl"
with open(tmp_pkl_file, "rb") as tmp_file_load:
    data_tuple = pkl.load(tmp_file_load)
    subword_vocab = pkl.load(tmp_file_load)
    idx_2_subword = pkl.load(tmp_file_load)
    subword_2_idx = pkl.load(tmp_file_load)

# Filter the dataset. #
filtered_data = []
for tmp_data in data_tuple:
    if len(tmp_data[0]) <= seq_encode \
        and len(tmp_data[1]) <= (seq_decode-1):
        filtered_data.append(tmp_data)
del tmp_data, data_tuple

data_tuple = filtered_data
vocab_size = len(subword_vocab)
print("Vocabulary Size:", str(vocab_size))
del filtered_data

num_data  = len(data_tuple)
SOS_token = subword_2_idx["<SOS>"]
EOS_token = subword_2_idx["<EOS>"]
PAD_token = subword_2_idx["<PAD>"]
UNK_token = subword_2_idx["<UNK>"]

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

print("Building the Transformer Model.")
start_time = time.time()

seq2seq_model = tf_transformer.Transformer(
    num_layers, num_heads, hidden_size, 
    ffwd_size, vocab_size, vocab_size, seq_encode, 
    seq_decode, rate1=0.0, rate2=1.0-prob_keep)
seq2seq_optim = tf.keras.optimizers.Adam(
    beta_1=0.9, beta_2=0.98, epsilon=1.0e-9)

elapsed_time = (time.time() - start_time) / 60
print("Transformer Model built", 
      "(" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    seq2seq_model=seq2seq_model, 
    seq2seq_optim=seq2seq_optim)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

if restore_flag:
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Model restored from {}".format(
            manager.latest_checkpoint))
    else:
        print("Error: No latest checkpoint found.")
    
    train_loss_df = pd.read_csv(train_loss_file)
    train_loss_list = [tuple(
        train_loss_df.iloc[x].values) \
            for x in range(len(train_loss_df))]
else:
    print("Training a new model.")
    train_loss_list = []

# Placeholders to store the batch data. #
tmp_input   = np.zeros(
    [batch_size, seq_encode], dtype=np.int32)
tmp_seq_out = np.zeros(
    [batch_size, seq_decode+1], dtype=np.int32)
tmp_test_in = np.zeros([1, seq_encode], dtype=np.int32)

n_iter = ckpt.step.numpy().astype(np.int32)
print("-" * 50)
print("Training the Transformer Keras Network", 
      "(" + str(n_iter), "iterations).")
print("Total of", str(num_data), "training samples.")
    
tot_loss = 0.0
start_tm = time.time()
while n_iter < maximum_iter:
    step_val = float(max(n_iter+1, warmup_steps))**(-0.5)
    learn_rate_val = float(hidden_size)**(-0.5) * step_val
    
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    tmp_input[:, :]   = PAD_token
    tmp_seq_out[:, :] = PAD_token
    tmp_seq_out[:, 0] = SOS_token
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_i_idx = data_tuple[tmp_index][0]
        tmp_o_idx = data_tuple[tmp_index][1]

        n_input  = len(tmp_i_idx)
        n_output = len(tmp_o_idx)
        n_decode = n_output + 1

        tmp_input[n_index, :n_input] = tmp_i_idx
        tmp_seq_out[n_index, 1:n_decode] = tmp_o_idx
        tmp_seq_out[n_index, n_decode] = EOS_token

    tmp_decode = tmp_seq_out[:, :-1]
    tmp_output = tmp_seq_out[:, 1:]
    
    tmp_loss = train_step(
        seq2seq_model, sub_batch_sz, 
        tmp_input, tmp_decode, tmp_output, 
        seq2seq_optim, learning_rate=learn_rate_val)
    
    n_iter += 1
    tot_loss += tmp_loss.numpy()
    ckpt.step.assign_add(1)

    if n_iter % display_step == 0:
        end_time = time.time()
        avg_loss = tot_loss / display_step
        tot_loss = 0.0
        elapsed_tm = (end_time - start_tm) / 60
        start_tm   = time.time()

        tmp_test_in[:, :] = PAD_token
        sample_id = np.random.choice(num_data, size=1)
        tmp_data  = data_tuple[sample_id[0]]
        
        tmp_i_idx = tmp_data[0]
        tmp_i_tok = bp_decode(tmp_i_idx, idx_2_subword)
        tmp_o_tok = bp_decode(tmp_data[1], idx_2_subword)
        
        n_input = len(tmp_i_idx)
        tmp_test_in[0, :n_input] = tmp_i_idx
        
        gen_ids = seq2seq_model.infer(
            tmp_test_in, SOS_token)
        gen_phrase = bp_decode(
            gen_ids.numpy()[0], idx_2_subword)

        print("Iteration", str(n_iter) + ":")
        print("Elapsed Time:", str(elapsed_tm) + " mins.")
        print("Average Loss:", str(avg_loss))
        print("Gradient Clip:", str(gradient_clip))
        print("Learning Rate:", str(seq2seq_optim.lr.numpy()))
        
        print("")
        print("Input Phrase:")
        print(" ".join(tmp_i_tok).replace("<", "").replace(">", ""))
        print("Generated Phrase:")
        print(" ".join(gen_phrase).replace("<", "").replace(">", ""))
        print("Actual Response:")
        print(" ".join(tmp_o_tok).replace("<", "").replace(">", ""))
        print("")
        
        # Save the training progress. #
        train_loss_list.append((n_iter, avg_loss))
        train_loss_df = pd.DataFrame(
            train_loss_list, columns=["n_iter", "xent_loss"])
        train_loss_df.to_csv(train_loss_file, index=False)
        
        # Save the model. #
        save_path = manager.save()
        print("Saved model to {}".format(save_path))
        print("-" * 50)
    
    if n_iter % cooling_step == 0:
        print("Cooling the CPU for 2 minutes.")
        time.sleep(120)
        print("Resuming training.")


