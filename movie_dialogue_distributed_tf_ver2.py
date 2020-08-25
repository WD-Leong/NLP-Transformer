
import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tf_ver2_transformer as tf_transformer

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Define the distributed training strategy. #
mirrored_strategy = tf.distribute.MirroredStrategy()

# Define the loss function. #
with mirrored_strategy.scope():
    def compute_losses(labels, logits, global_batch_size):
        per_eg_loss  = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits), axis=1)
        
        tmp_avg_loss = tf.nn.compute_average_loss(
            per_eg_loss, global_batch_size=global_batch_size)
        return tmp_avg_loss

# Define the weight update step. #
#@tf.function
def train_step(
    model, global_batch_size, 
    x_encode, x_decode, x_output, 
    optimizer, learning_rate=1.0e-3, grad_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    global_batch_size = tf.cast(
        global_batch_size, tf.float32)
    with tf.GradientTape() as grad_tape:
        output_logits = model(x_encode, x_decode)
        tmp_dist_loss = compute_losses(
            x_output, output_logits, global_batch_size)
    
    tmp_gradients = grad_tape.gradient(
        tmp_dist_loss, model.trainable_variables)
    
    clipped_gradients, _ = \
        tf.clip_by_global_norm(tmp_gradients, grad_clip)
    optimizer.apply_gradients(
        zip(clipped_gradients, model.trainable_variables))
    return tmp_dist_loss

def distributed_train_step(
    mirrored_strategy, model, 
    global_batch_size, tmp_input_tuple, 
    optimizer, learning_rate=1.0e-3, grad_clip=1.0):
    x_encode = tmp_input_tuple[0]
    x_decode = tmp_input_tuple[1]
    x_output = tmp_input_tuple[2]
    
    distributed_args = (
        model, global_batch_size, 
        x_encode, x_decode, x_output, 
        optimizer, learning_rate, grad_clip,)
    per_replica_loss = \
        mirrored_strategy.run(
            train_step, args=distributed_args)
    return mirrored_strategy.reduce(
        tf.distribute.ReduceOp.SUM, 
        per_replica_loss, axis=None)

# Model Parameters. #
batch_size = 256
seq_encode = 50
seq_decode = 51

num_layers  = 6
num_heads   = 16
prob_keep   = 0.9
hidden_size = 1024
ffwd_size   = 4*hidden_size

initial_lr    = 0.001
gradient_clip = 1.00
maximum_epoch = 500
maximum_steps = 500000
restore_flag  = True
display_step  = 100
save_step     = 1000
cooling_step  = 1000
warmup_steps  = 5000
anneal_step   = 2000
anneal_rate   = 0.75

tmp_path = "/home/"
model_ckpt_dir  = tmp_path +\
    "TF_Models/transformer_seq2seq_distributed"
train_loss_file = tmp_path +\
    "Codes/train_loss_transformer_distributed.csv"

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

print("Building the Transformer Model.")
start_time = time.time()

with mirrored_strategy.scope():
    seq2seq_model = tf_transformer.TransformerNetwork(
        num_layers, num_heads, hidden_size, ffwd_size, 
        vocab_size, vocab_size, seq_encode, seq_decode, 
        embed_size=hidden_size, p_keep=prob_keep)
    seq2seq_optimizer = tf.keras.optimizers.Adam(
        beta_1=0.9, beta_2=0.98, epsilon=1.0e-9)

    elapsed_time = (time.time() - start_time) / 60
    print("Transformer Model built (" + str(elapsed_time) + " mins).")
    
    # Create the model checkpoint. #
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(0), 
        epoch=tf.Variable(0), 
        seq2seq_model=seq2seq_model, 
        seq2seq_optimizer=seq2seq_optimizer)
    
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
            train_loss_df.iloc[x].values) for x in range(len(train_loss_df))]
    else:
        print("Training a new model.")
        train_loss_list = []

# Create the TF data pipeline. #
tmp_seq_out = np.zeros([seq_decode+1], dtype=np.int32)

tmp_qns = np.zeros(
    [len(data_tuple), seq_encode], dtype=np.int32)
tmp_dec = np.zeros(
    [len(data_tuple), seq_decode], dtype=np.int32)
tmp_out = np.zeros(
    [len(data_tuple), seq_decode], dtype=np.int32)
for n_tuple in range(len(data_tuple)):
    tmp_seq_out[:] = PAD_token
    tmp_seq_out[0] = SOS_token
    
    tmp_tuple = data_tuple[n_tuple]
    tmp_i_tok = tmp_tuple[0].split(" ")
    tmp_o_tok = tmp_tuple[1].split(" ")
    
    tmp_i_idx = [word2idx.get(
        x, UNK_token) for x in tmp_i_tok]
    tmp_o_idx = [word2idx.get(
        x, UNK_token) for x in tmp_o_tok]
    
    n_input  = len(tmp_i_idx)
    n_output = len(tmp_o_idx)
    n_decode = n_output + 1
    
    tmp_seq_out[1:n_decode] = tmp_o_idx
    tmp_seq_out[n_decode]   = EOS_token
    tmp_qns[n_tuple, :n_input] = tmp_i_idx
    
    tmp_dec[n_tuple, :] = tmp_seq_out[:-1]
    tmp_out[n_tuple, :] = tmp_seq_out[1:]

# Distribute the dataset. #
train_dataset = tf.data.Dataset.from_tensor_slices(
    (tmp_qns, tmp_dec, tmp_out)).batch(batch_size)
tr_dist_data  = \
    mirrored_strategy.experimental_distribute_dataset(train_dataset)

# Placeholders to store the batch data. #
tmp_test_in = np.zeros([1, seq_encode], dtype=np.int32)
tmp_test_dec = SOS_token * np.ones([1, 1], dtype=np.int32)

n_iter  = ckpt.step.numpy().astype(np.int32)
n_epoch = ckpt.epoch.numpy().astype(np.int32)

print("-" * 50)
print("Training the Transformer Network", 
      "(" + str(n_iter), "iterations).")

tot_loss = 0.0
start_tm = time.time()
while n_epoch < maximum_epoch:
    step_val = float(max(n_iter+1, warmup_steps))**(-0.5)
    lr_value = max(
       1.0e-6, float(hidden_size)**(-0.5) * step_val)
    
    for tmp_tuple in tr_dist_data:
        tmp_loss = distributed_train_step(
            mirrored_strategy, seq2seq_model, batch_size, 
            tmp_tuple, seq2seq_optimizer, learning_rate=lr_value)
        tot_loss += tmp_loss.numpy()
        
        # Increment the step. #
        n_iter += 1
        ckpt.step.assign_add(1)
        
        if n_iter % display_step == 0:
            end_time = time.time()
            avg_loss = tot_loss / display_step
            tot_loss = 0.0
            elapsed_tm = (end_time - start_tm) / 60
            start_tm   = time.time()
            
            tmp_test_in[:, :] = PAD_token
            sample_id = np.random.choice(num_data, size=1)
            sw_tokens = wiki_sw_articles[sample_id[0]]
            
            n_input = len(sw_tokens)
            tmp_test_in[0, :n_input] = tmp_i_idx
            
            gen_ids = seq2seq_model.infer(tmp_test_in, tmp_test_dec)
            gen_phrase = [idx2word[x] for x in gen_ids.numpy()[0]]
            gen_phrase = " ".join(gen_phrase)
            
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
            print("")
        
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
            print("Saved model to {}".format(save_path))
            print("-" * 50)
        
        if n_iter > maximum_steps:
            break
    # Increment the epoch. #
    n_epoch += 1
    ckpt.epoch.assign_add(1)


