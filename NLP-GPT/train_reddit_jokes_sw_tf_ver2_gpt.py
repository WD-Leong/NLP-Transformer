import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tf_ver2_gpt_keras as gpt

# Custom functions. #
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

# Define the weight update step for multiple sub-batches. #
def sub_batch_train_step(
    model, sub_batch_sz, x_encode, x_output, optimizer, 
    learning_rate=1.0e-3, grad_clip=1.0, foc_loss=False):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_encode.shape[0]
    if batch_size <= sub_batch_sz:
        sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        sub_batch = int(batch_size / sub_batch_sz)
    else:
        sub_batch = int(batch_size / sub_batch_sz) + 1
    
    model_params  = model.trainable_variables
    acc_gradients = [tf.zeros_like(var) for var in model_params]
    
    tot_losses = 0.0
    for n_sub in range(sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_encode = x_encode[id_st:id_en, :]
        tmp_output = x_output[id_st:id_en, :]
        
        with tf.GradientTape() as grad_tape:
            output_logits = model(tmp_encode, training=True)
            
            tmp_losses = tf.reduce_sum(tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_output, logits=output_logits), axis=1))
        
        # Accumulate the gradients. #
        tot_losses += tmp_losses
        tmp_gradients = grad_tape.gradient(
            tmp_losses, model_params)
        acc_gradients = [tf.add(
            acc_grad, grad) for acc_grad, grad \
                in zip(acc_gradients, tmp_gradients)]
    
    # Update using the optimizer. #
    avg_losses = tot_losses / batch_size
    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    
    clipped_gradients, _ = tf.clip_by_global_norm(
        acc_gradients, grad_clip)
    optimizer.apply_gradients(
        zip(clipped_gradients, model_params))
    return avg_losses

# Model Parameters. #
batch_size = 256
sub_batch  = 64
seq_length = 31
num_heads  = 4
num_layers = 3

gradient_clip = 1.00
maximum_iter  = 10000
restore_flag  = False
save_step     = 500
warmup_steps  = 5000
display_step  = 100
anneal_step   = 2500
anneal_rate   = 0.75

prob_keep = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size
warmup_flag = True
cooling_step = 1000

model_ckpt_dir  = "../TF_Models/gpt_keras_subword_reddit"
train_loss_file = "train_loss_gpt_keras_subword_reddit.csv"

# Load the data. #
tmp_pkl_file = "reddit_jokes_subword.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    full_data = pkl.load(tmp_load_file)
    idx2subword = pkl.load(tmp_load_file)
    subword2idx = pkl.load(tmp_load_file)

vocab_size = len(subword2idx)
print("Vocabulary Size:", str(vocab_size)+".")

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

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

# Build the GPT. #
print("Building the GPT Keras Model.")
start_time = time.time()

gpt_model = gpt.GPTDecoder(
    num_layers, num_heads, hidden_size, 
    ffwd_size, vocab_size, seq_length, 
    rate1=0.0, rate2=1.0-prob_keep)
gpt_optimizer = tf.keras.optimizers.Adam(
    beta_1=0.9, beta_2=0.98, epsilon=1.0e-9)

elapsed_time = (time.time() - start_time) / 60
print("GPT Keras Model Built", 
      "(" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    gpt_model=gpt_model, 
    gpt_optimizer=gpt_optimizer)

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

# Train the Transformer model. #
tmp_out_seq = np.zeros(
    [batch_size, seq_length+1], dtype=np.int32)
tmp_test_in = np.zeros([1, seq_length], dtype=np.int32)

# Warmup learning schedule. #
n_iter = ckpt.step.numpy().astype(np.int32)
if warmup_flag:
    #step_min = min(
    #    (n_iter+1)**(-0.5), (n_iter+1)*warmup_steps**(-1.5))
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
            anneal_factor = np.power(
                anneal_rate, int(n_iter / anneal_step))
            learning_rate = \
                max(anneal_factor*initial_lr, 1.0e-6)
    
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
        del tmp_p_idx
    
    # Set the training data. #
    tmp_input  = tmp_out_seq[:, :-1]
    tmp_output = tmp_out_seq[:, 1:]
    
    tmp_loss = sub_batch_train_step(
        gpt_model, sub_batch, tmp_input, tmp_output, 
        gpt_optimizer, learning_rate=learning_rate)

    n_iter += 1
    ckpt.step.assign_add(1)
    
    tot_loss += tmp_loss.numpy()
    if n_iter % display_step == 0:
        end_tm = time.time()
        
        avg_loss = tot_loss / display_step
        tot_loss = 0.0
        elapsed_tm = (end_tm - start_tm) / 60
        
        tmp_test_in[:, :] = PAD_token
        sample_test = np.random.choice(num_data, size=1)
        tmp_p_index = tmp_data[sample_test[0]]
        
        in_phrase = bp_decode(tmp_p_index, idx2subword)
        in_phrase = " ".join(in_phrase).replace(
            "<", "").replace(">", "")
        
        n_tokens = len(tmp_p_index)
        n_sample = min(np.random.randint(
            1, high=n_tokens-1), int(n_tokens/2))
        tmp_test_in[0, :n_tokens] = tmp_p_index
        
        tmp_infer = gpt_model.infer(tmp_test_in[:, :n_sample])
        del sample_test, n_tokens
        
        gen_phrase = bp_decode(
            tmp_infer[0].numpy(), idx2subword)
        gen_phrase = " ".join(gen_phrase).replace(
            "<", "").replace(">", "")
        
        test_phrase = bp_decode(
            tmp_p_index[:n_sample], idx2subword)
        test_phrase = " ".join(test_phrase).replace(
            "<", "").replace(">", "")
        del tmp_p_index
        
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
        save_path = manager.save()
        print("Saved model to {}".format(save_path))
        
        tmp_df_losses = pd.DataFrame(
            train_loss_list, columns=["n_iter", "xent_loss"])
        tmp_df_losses.to_csv(train_loss_file, index=False)
        del tmp_df_losses
    
    # Cool the GPU. #
    if n_iter % cooling_step == 0:
        print("Cooling GPU for 2 minutes.")
        time.sleep(120)
        print("Resume Training.")
        print("-" * 50)

