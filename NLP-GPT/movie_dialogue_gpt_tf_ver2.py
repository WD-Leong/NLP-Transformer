
import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tf_ver2_gpt as tf_gpt

# Define the weight update step. #
#@tf.function
def train_step(
    model, x_input, x_output, optimizer, 
    learning_rate=1.0e-3, grad_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    with tf.GradientTape() as grad_tape:
        output_logits = model(x_input)
        
        tmp_losses = tf.reduce_mean(tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=x_output, logits=output_logits), axis=1))
    
    tmp_gradients = \
        grad_tape.gradient(tmp_losses, model.trainable_variables)
    
    clipped_gradients, _ = \
        tf.clip_by_global_norm(tmp_gradients, grad_clip)
    optimizer.apply_gradients(
        zip(clipped_gradients, model.trainable_variables))
    return tmp_losses

# Model Parameters. #
batch_size = 128
seq_length = 101

num_layers  = 6
num_heads   = 16
prob_keep   = 0.9
hidden_size = 1024
ffwd_size   = 4*hidden_size

initial_lr    = 0.001
gradient_clip = 1.00
maximum_iter  = 500000
restore_flag  = True
display_step  = 100
save_step     = 1000
cooling_step  = 1000
warmup_steps  = 5000
anneal_step   = 2000
anneal_rate   = 0.75

tmp_path = "/home/"
model_ckpt_dir  = tmp_path +\
    "TF_Models/dialogue_gpt"
train_loss_file = tmp_path +\
    "Codes/dialogue_train_loss_gpt.csv"

tmp_pkl_file = tmp_path + "Data/movie_dialogues.pkl"
with open(tmp_pkl_file, "rb") as tmp_file_load:
    data_tuple = pkl.load(tmp_file_load)
    idx2word = pkl.load(tmp_file_load)
    word2idx = pkl.load(tmp_file_load)

vocab_size = len(word2idx)
print("Vocabulary Size:", str(vocab_size))

num_data  = len(data_tuple)
SOS_token = word2idx["SOS"]
EOS_token = word2idx["EOS"]
PAD_token = word2idx["PAD"]
UNK_token = word2idx["UNK"]

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

print("Building the Transformer Model.")
start_time = time.time()

seq2seq_model = tf_gpt.GPT_Network(
    num_layers, num_heads, hidden_size, 
    ffwd_size, vocab_size, seq_length, 
    embed_size=hidden_size, p_keep=prob_keep)
seq2seq_optimizer = tf.keras.optimizers.Adam(
    beta_1=0.9, beta_2=0.98, epsilon=1.0e-9)

elapsed_time = (time.time() - start_time) / 60
print("GPT Model built (" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    seq2seq_model=seq2seq_model, 
    seq2seq_optimizer=seq2seq_optimizer)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

if restore_flag:
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Model restored from {}".format(manager.latest_checkpoint))
    else:
        print("Error: No latest checkpoint found.")
    
    train_loss_df = pd.read_csv(train_loss_file)
    train_loss_list = [tuple(
        train_loss_df.iloc[x].values) for x in range(len(train_loss_df))]
else:
    print("Training a new model.")
    train_loss_list = []

# Placeholders to store the batch data. #
tmp_sequence = np.zeros(
    [batch_size, seq_length+1], dtype=np.int32)
tmp_test_in  = np.zeros([1, seq_length], dtype=np.int32)

n_iter = ckpt.step.numpy().astype(np.int32)
print("-" * 50)
print("Training the Transformer Network", 
      "(" + str(n_iter), "iterations).")

tot_loss = 0.0
start_tm = time.time()
while n_iter < maximum_iter:
    #annealing_val  = anneal_rate ** ((n_iter+1) // anneal_step)
    #learn_rate_val = max(annealing_val * initial_lr, 1.0e-5)

    step_val = float(max(n_iter+1, warmup_steps))**(-0.5)
    lr_iter  = float(hidden_size)**(-0.5) * step_val
    learn_rate_val = max(lr_iter, 1.0e-6)
    
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    tmp_sequence[:, :] = PAD_token
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_i_tok = data_tuple[tmp_index][0].split(" ")
        tmp_o_tok = data_tuple[tmp_index][1].split(" ")

        tmp_i_idx  = [word2idx.get(x, UNK_token) for x in tmp_i_tok]
        tmp_o_idx  = [word2idx.get(x, UNK_token) for x in tmp_o_tok]
        tmp_seq_id = tmp_i_idx + [SOS_token] + tmp_o_idx
        n_sequence = len(tmp_seq_id)
        
        tmp_sequence[n_index, :n_sequence] = tmp_seq_id
        tmp_sequence[n_index, n_sequence]  = EOS_token
    
    tmp_input  = tmp_sequence[:, :-1]
    tmp_output = tmp_sequence[:, 1:]
    
    tmp_loss = train_step(
        seq2seq_model, tmp_input, tmp_output, 
        seq2seq_optimizer, learning_rate=learn_rate_val)
    
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

        tmp_i_tok = tmp_data[0].split(" ")
        tmp_o_tok = tmp_data[1].split(" ")
        tmp_i_idx = [word2idx.get(x, UNK_token) for x in tmp_i_tok]

        n_input  = len(tmp_i_idx)
        n_seq_in = n_input + 1
        tmp_test_in[0, :n_input] = tmp_i_idx
        tmp_test_in[0, n_input]  = SOS_token
        
        gen_ids = \
            seq2seq_model.infer(tmp_test_in[:, :n_seq_in])
        full_phrase = [idx2word[x] for x in gen_ids.numpy()[0]]
        gen_phrase  = " ".join(full_phrase[n_input:])

        print("Iteration", str(n_iter) + ":")
        print("Elapsed Time:", str(elapsed_tm) + " mins.")
        print("Average Loss:", str(avg_loss))
        print("Gradient Clip:", str(gradient_clip))
        print("Learning Rate:", str(seq2seq_optimizer.lr.numpy()))

        print("")
        print("Input Phrase:")
        print(" ".join([idx2word[x] for x in tmp_i_idx]))
        print("Generated Phrase:")
        print(gen_phrase)
        print("Actual Response:")
        print(tmp_data[1])
        
        if n_iter % save_step != 0:
            print("-" * 50)
    
    if n_iter % save_step == 0:
        # Save the training progress. #
        train_loss_list.append((n_iter, avg_loss))
        train_loss_df = pd.DataFrame(
            train_loss_list, columns=["n_iter", "xent_loss"])
        train_loss_df.to_csv(train_loss_file, index=False)
        
        # Save the model. #
        save_path = manager.save()
        print("")
        print("Saved model to {}".format(save_path))
        print("-" * 50)


