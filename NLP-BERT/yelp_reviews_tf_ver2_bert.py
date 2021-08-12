import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_bert_module as bert
from sklearn.metrics import classification_report

# Define the weight update step for multiple sub-batches. #
def sub_batch_train_step(
    model, sub_batch_sz, 
    x_encode, x_output, optimizer, 
    learning_rate=1.0e-3, grad_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_encode.shape[0]
    if batch_size <= sub_batch_sz:
        sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        sub_batch = int(batch_size / sub_batch_sz)
    else:
        sub_batch = int(batch_size / sub_batch_sz) + 1
    
    model_params  = model.trainable_variables
    acc_gradients = [
        tf.zeros_like(var) for var in model_params]
    
    tot_losses = 0.0
    for n_sub in range(sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_output = x_output[id_st:id_en]
        tmp_encode = x_encode[id_st:id_en, :]
        
        with tf.GradientTape() as grad_tape:
            output_logits = model(
                tmp_encode, training=True)[0]
            
            tmp_losses = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_output, logits=output_logits))
        
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
batch_test = 128
sub_batch  = 64
seq_length = 200
num_heads  = 4
num_layers = 3

gradient_clip = 1.00
maximum_iter  = 5500
restore_flag  = True
save_step     = 100
warmup_steps  = 5000
display_step  = 50
anneal_step   = 2500
anneal_rate   = 0.75

grad_clip  = 1.0
prob_noise = 0.0
prob_keep  = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size
warmup_flag = True
cooling_step = 100

tmp_path = "C:/Users/admin/Desktop/Codes/"
model_ckpt_dir  = "../TF_Models/yelp_bert"
train_loss_file = "train_loss_yelp_bert.csv"

# Load the data. #
tmp_pkl_file = "C:/Users/admin/Desktop/Data/Yelp/"
tmp_pkl_file += "yelp_review_polarity_csv/test_bert_data.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    test_data = pkl.load(tmp_load_file)
    word_vocab = pkl.load(tmp_load_file)
    idx_2_word = pkl.load(tmp_load_file)
    word_2_idx = pkl.load(tmp_load_file)
del word_vocab, word_2_idx, idx_2_word, tmp_pkl_file

tmp_pkl_file = "C:/Users/admin/Desktop/Data/Yelp/"
tmp_pkl_file += "yelp_review_polarity_csv/train_bert_data.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    train_data = pkl.load(tmp_load_file)
    word_vocab = pkl.load(tmp_load_file)
    idx_2_word = pkl.load(tmp_load_file)
    word_2_idx = pkl.load(tmp_load_file)

label_dict = list(sorted(
    list(set([x[0] for x in train_data]))))
vocab_size = len(word_vocab)
print("Vocabulary Size:", str(vocab_size) + ".")

# Note that CLS token acts as SOS token for GPT. #
num_test  = len(test_data)
num_data  = len(train_data)
num_class = len(label_dict)

CLS_token = word_2_idx["CLS"]
EOS_token = word_2_idx["EOS"]
PAD_token = word_2_idx["PAD"]
UNK_token = word_2_idx["UNK"]
TRU_token = word_2_idx["TRUNC"]
print("Total of", str(len(train_data)), "rows loaded.")

if num_test <= batch_test:
    n_val_batches = 1
elif num_test % batch_test == 0:
    n_val_batches = int(num_test / batch_test)
else:
    n_val_batches = int(num_test / batch_test) + 1

# Extract and normalise the test labels. #
test_labels = np.array([
    (int(x[0]) - 1) for x in test_data])

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Build the BERT Model. #
print("Building the BERT Model.")
start_time = time.time()

bert_model = bert.BERT_Classifier(
    num_class, num_layers, num_heads, 
    hidden_size, ffwd_size, word_2_idx, 
    seq_length + 3, p_keep=prob_keep)
bert_optimizer = tfa.optimizers.AdamW(
    weight_decay=1.0e-4)

elapsed_time = (time.time()-start_time) / 60
print("BERT Model Built", 
      "(" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    bert_model=bert_model, 
    bert_optimizer=bert_optimizer)

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
tmp_in_seq  = np.zeros(
    [batch_size, seq_length+3], dtype=np.int32)
tmp_out_lab = np.zeros([batch_size], dtype=np.int32)

# Warmup learning schedule. #
n_iter = ckpt.step.numpy().astype(np.int32)
if warmup_flag:
    step_min = float(max(n_iter, warmup_steps))**(-0.5)
    learning_rate = float(hidden_size)**(-0.5) * step_min
else:
    initial_lr = 0.001
    anneal_pow = int(n_iter / anneal_step)
    learning_rate = max(np.power(
        anneal_rate, anneal_pow)*initial_lr, 1.0e-5)

print("-" * 50)
print("Training the BERT Network", 
      "(" + str(n_iter) + " iterations).")
print(str(num_data), "training samples.")
print(str(num_test), "test data samples.")
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
            anneal_pow = int(n_iter / anneal_step)
            learning_rate = max(np.power(
                anneal_rate, anneal_pow)*initial_lr, 1.0e-4)
    
    # Select a sample from the data. #
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    tmp_out_lab[:] = 0
    tmp_in_seq[:, :] = PAD_token
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_label = train_data[tmp_index][0]
        tmp_i_tok = train_data[tmp_index][1]
        
        # Truncate the sequence if it exceeds the maximum #
        # sequence length. To make the model more robust, #
        # a random permutation of the sequence is taken.  #
        if len(tmp_i_tok) > seq_length:
            idx_permute = list(sorted(
                list(np.random.permutation(
                    len(tmp_i_tok))[:seq_length])))
            
            tmp_i_idx = [word_2_idx.get(
                tmp_i_tok[x], UNK_token) for x in idx_permute]
            
            # Add the TRUNCATE token. #
            n_input  = len(tmp_i_idx)
            n_decode = n_input + 3
            del idx_permute
        else:
            tmp_i_idx = [word_2_idx.get(
                x, UNK_token) for x in tmp_i_tok]
            n_input  = len(tmp_i_idx)
            n_decode = n_input + 2
        
        tmp_label = int(tmp_label) - 1
        in_noisy  = [UNK_token] * n_input
        if prob_noise > 0:
            tmp_flag = np.random.binomial(
                1, prob_noise, size=n_input)
            in_noise = [
                tmp_i_idx[x] if tmp_flag[x] == 1 else \
                    in_noisy[x] for x in range(n_input)]
            
            if len(tmp_i_tok) > seq_length:
                tmp_p_idx = \
                    [CLS_token] + in_noise + [TRU_token, EOS_token]
            else:
                tmp_p_idx = [CLS_token] + in_noise + [EOS_token]
        else:
            if len(tmp_i_tok) > seq_length:
                tmp_p_idx = \
                    [CLS_token] + tmp_i_idx + [TRU_token, EOS_token]
            else:
                tmp_p_idx = [CLS_token] + tmp_i_idx + [EOS_token]
        
        tmp_out_lab[n_index] = tmp_label
        tmp_in_seq[n_index, :n_decode] = tmp_p_idx
    
    # Set the training data. #
    tmp_input  = tmp_in_seq
    tmp_output = tmp_out_lab
    
    tmp_loss = sub_batch_train_step(
        bert_model, sub_batch, 
        tmp_input, tmp_output, bert_optimizer, 
        learning_rate=learning_rate, grad_clip=grad_clip)
    
    n_iter += 1
    ckpt.step.assign_add(1)
    
    tot_loss += tmp_loss.numpy()
    if n_iter % display_step == 0:
        # For simplicity, get the test accuracy #
        # instead of validation accuracy.       #
        pred_labels = []
        for n_val_batch in range(n_val_batches):
            id_st = n_val_batch * batch_test
            id_en = (n_val_batch+1) * batch_test
            
            if n_val_batch == (n_val_batches-1):
                curr_batch = num_test - id_st
            else:
                curr_batch = batch_test
            
            tmp_test_tokens = np.zeros(
                [curr_batch, seq_length+3], dtype=np.int32)
            
            tmp_test_tokens[:, :] = PAD_token
            tmp_test_tokens[:, 0] = CLS_token
            
            for tmp_n in range(curr_batch):
                tmp_input = [
                    word_2_idx.get(x, UNK_token) for \
                        x in test_data[id_st+tmp_n][1]]
                
                # Truncate if the length is longer. #
                n_input  = len(tmp_input)
                n_decode = n_input + 1
                if n_input > seq_length:
                    tru_toks = tmp_input[:seq_length]
                    tmp_toks = \
                        [CLS_token] + tru_toks + [TRU_token, EOS_token]
                    del tru_toks
                else:
                    tmp_toks = [CLS_token] + tmp_input + [EOS_token]
                n_decode = len(tmp_toks)
                
                tmp_test_tokens[tmp_n, :n_decode] = tmp_toks
                del tmp_toks, tmp_input, n_input, n_decode
            
            # Perform inference. #
            tmp_pred_labels = \
                bert_model.infer(tmp_test_tokens).numpy()
            pred_labels.append(tmp_pred_labels)
            del tmp_test_tokens
        
        # Concatenate the predicted labels. #
        pred_labels = np.concatenate(
            tuple(pred_labels), axis=0)
        del tmp_pred_labels
        
        # Compute the accuracy. #
        accuracy = np.sum(np.where(
            pred_labels == test_labels, 1, 0)) / num_test
        del pred_labels
        
        end_tm = time.time()
        avg_loss = tot_loss / display_step
        tot_loss = 0.0
        elapsed_tm = (end_tm - start_tm) / 60
        
        print("Iteration", str(n_iter) + ".")
        print("Elapsed Time:", str(elapsed_tm), "mins.")
        print("Gradient Clip:", str(gradient_clip) + ".")
        print("Learning Rate:", str(learning_rate) + ".")
        print("Average Loss:", str(avg_loss) + ".")
        print("Test Accuracy:", str(round(accuracy*100, 2)) + "%.")
        
        train_loss_list.append((n_iter, avg_loss, accuracy))
        start_tm = time.time()
        print("-" * 50)
    
    # Model checkpoint. #
    if n_iter % save_step == 0:
        # Save the model. #
        save_path = manager.save()
        print("Saved model to {}".format(save_path))
        
        tmp_df_column = ["n_iter", "xent_loss", "test_acc"]
        tmp_df_losses = pd.DataFrame(
            train_loss_list, columns=tmp_df_column)
        tmp_df_losses.to_csv(train_loss_file, index=False)
        del tmp_df_losses
    
    # Cool the GPU. #
    if n_iter % cooling_step == 0:
        print("Cooling GPU for 2 minutes.")
        time.sleep(120)
        print("Resume Training.")
        print("-" * 50)

# Print the evaluation report. #
pred_labels = []
for n_val_batch in range(n_val_batches):
    id_st = n_val_batch * batch_test
    id_en = (n_val_batch+1) * batch_test
    
    if n_val_batch == (n_val_batches-1):
        curr_batch = num_test - id_st
    else:
        curr_batch = batch_test
    
    tmp_test_tokens = np.zeros(
        [curr_batch, seq_length+3], dtype=np.int32)
    
    tmp_test_tokens[:, :] = PAD_token
    tmp_test_tokens[:, 0] = CLS_token
    
    for tmp_n in range(curr_batch):
        tmp_input = [
            word_2_idx.get(x, UNK_token) for \
                x in test_data[id_st+tmp_n][1]]
        
        # Truncate if the length is longer. #
        n_input  = len(tmp_input)
        n_decode = n_input + 1
        if n_input > seq_length:
            tru_toks = tmp_input[:seq_length]
            tmp_toks = \
                [CLS_token] + tru_toks + [TRU_token, EOS_token]
            del tru_toks
        else:
            tmp_toks = [CLS_token] + tmp_input + [EOS_token]
        n_decode = len(tmp_toks)
        
        tmp_test_tokens[tmp_n, :n_decode] = tmp_toks
        del tmp_toks, tmp_input, n_input, n_decode
    
    # Perform inference. #
    tmp_pred_labels = \
        bert_model.infer(tmp_test_tokens).numpy()
    pred_labels.append(tmp_pred_labels)
    del tmp_test_tokens

# Concatenate the predicted labels. #
print("Generating evaluation report.")

pred_labels = np.concatenate(
    tuple(pred_labels), axis=0)
del tmp_pred_labels

pred_report = classification_report(
    test_labels, pred_labels)
with open("yelp_validation_report.txt", "w") as tmp_write:
    tmp_write.write(pred_report)
print(pred_report)
