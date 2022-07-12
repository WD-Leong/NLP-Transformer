import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_bert_keras as bert

from bert_utils import (
    sub_batch_train_step, generate_input)
from sklearn.metrics import classification_report

# Load the data. #
print("Loading and processing the data.")

tmp_path = "/home/Data/emotion_dataset/"
tmp_file = tmp_path + "emotion_word_bert.pkl"
with open(tmp_file, "rb") as tmp_file_load:
    train_data = pkl.load(tmp_file_load)
    valid_data = pkl.load(tmp_file_load)
    
    word_vocab = pkl.load(tmp_file_load)
    idx_2_word = pkl.load(tmp_file_load)
    word_2_idx = pkl.load(tmp_file_load)
    
    labels_2_idx = pkl.load(tmp_file_load)
    idx_2_labels = pkl.load(tmp_file_load)

vocab_size = len(word_vocab)
print("Vocabulary Size:", str(vocab_size) + ".")

# For pre-training. #
special_token = ["CLS", "EOS", "PAD", "UNK", "MSK", "TRU"]
replace_vocab = [
    word_2_idx[x] for x in \
        word_vocab if x not in special_token]

# Extract a sample of the validation data. #
tuple_valid = valid_data
del valid_data

num_approx = 2500
valid_data = [x for x in tuple_valid[:num_approx]]

CLS_token = word_2_idx["CLS"]
EOS_token = word_2_idx["EOS"]
PAD_token = word_2_idx["PAD"]
UNK_token = word_2_idx["UNK"]
MSK_token = word_2_idx["MSK"]
TRU_token = word_2_idx["TRU"]

# Parameters. #
p_keep = 0.90
p_mask = 0.15
grad_clip = 1.00

num_layers = 3
num_heads  = 4
n_classes  = len(labels_2_idx)

steps_max  = 10000
batch_size = 256
sub_batch  = 64
batch_test = 64

seq_length  = 30
hidden_size = 256
ffwd_size   = 4*hidden_size

warmup_steps = 2500
cooling_step = 250
display_step = 50
restore_flag = True
finetune_step = 5000
final_tunings = 7500

model_ckpt_dir  = "Model/bert_emotion_word_model"
train_loss_file = "bert_emotion_word_losses.csv"

# Define the classifier model. #
bert_model = bert.BERTClassifier(
    n_classes, num_layers, 
    num_heads, hidden_size, ffwd_size, 
    vocab_size, seq_length+2, rate=1.0-p_keep)
bert_optim = tfa.optimizers.AdamW(
    weight_decay=1.0e-4)

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    bert_model=bert_model, 
    bert_optim=bert_optim)

# Print the model summary. #
tmp_zero = np.zeros(
    [sub_batch, seq_length+2], dtype=np.int32)
tmp_pred = bert_model(tmp_zero, training=True)[0]

print(bert_model.summary())
del tmp_zero, tmp_pred

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
    del train_loss_df
else:
    print("Training a new model.")
    train_loss_list = []

# Extract the label indices for validation. #
y_valid = [
    labels_2_idx[x] for x, y in valid_data]
target_labels = [
    idx_2_labels[x] for x in range(n_classes)]

n_iter = ckpt.step.numpy().astype(np.int32)
num_train = len(train_data)
num_valid = len(valid_data)

if num_valid <= batch_test:
    n_val_batches = 1
elif num_valid % batch_test == 0:
    n_val_batches = int(num_valid / batch_test)
else:
    n_val_batches = int(num_valid / batch_test) + 1

# Train the model. #
print("Training BERT Model.")
print("(" + str(n_iter), "iterations.)")
print(str(num_train), "training data.")
print(str(num_valid), "validation data.")
print("-" * 50)

tot_loss = 0.0
tmp_in_seq  = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_in_mask = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_out_seq = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_out_lab = np.zeros([batch_size], dtype=np.int32)

start_tm = time.time()
while n_iter < steps_max:
    # Constant warmup rate. #
    step_val = float(max(n_iter+1, warmup_steps))**(-0.5)
    learn_rate_val = float(hidden_size)**(-0.5) * step_val
    
    if n_iter >= finetune_step:
        cls_flag = True
        if n_iter < final_tunings:
            learn_rate_val = 1.0e-4
        else:
            learn_rate_val = 1.0e-5
    else:
        cls_flag = False
        batch_size = 256
        sub_batch  = 64
    
    batch_sample = np.random.choice(
        num_train, size=batch_size, replace=False)
    
    # Set the mask to be ones to let it learn EOS #
    # and PAD token embeddings.                   # 
    tmp_out_lab[:] = 0
    tmp_in_mask[:, :] = 1.0
    tmp_in_seq[:, :]  = PAD_token
    tmp_out_seq[:, :] = PAD_token
    
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_label = labels_2_idx[train_data[tmp_index][0]]
        tmp_i_tok = [
            x for x in train_data[tmp_index][1].split(" ")]
        
        tmp_tuple = generate_input(
            tmp_i_tok, word_2_idx, p_mask, 
            seq_length, replace_vocab, CLS_token, 
            EOS_token, MSK_token, TRU_token, UNK_token)
        
        n_input = len(tmp_tuple[0])
        tmp_out_lab[n_index] = tmp_label
        tmp_in_seq[n_index, :n_input]  = tmp_tuple[0]
        tmp_out_seq[n_index, :n_input] = tmp_tuple[1]
        tmp_in_mask[n_index, :n_input] = tmp_tuple[2]
    
    tmp_loss = sub_batch_train_step(
        bert_model, sub_batch, tmp_in_mask, 
        tmp_in_seq, tmp_out_seq, tmp_out_lab, 
        bert_optim, cls_flag=cls_flag, reg_emb=0.01, 
        learning_rate=learn_rate_val, grad_clip=grad_clip)
    
    # Increment the step. #
    n_iter += 1
    ckpt.step.assign_add(1)
    
    tot_loss += tmp_loss.numpy()
    if n_iter % display_step == 0:
        # Get the validation accuracy. #
        pred_labels = []
        test_sample = np.random.choice(num_valid)
        
        test_flag = False
        for n_val_batch in range(n_val_batches):
            id_st = n_val_batch * batch_test
            if n_val_batch == (n_val_batches-1):
                id_en = num_valid
            else:
                id_en = (n_val_batch+1) * batch_test
            curr_batch = id_en - id_st
            
            x_valid = np.zeros(
                [curr_batch, seq_length+2], dtype=np.int32)
            
            x_valid[:, :] = PAD_token
            for n_tmp in range(curr_batch):
                curr_idx = id_st + n_tmp
                
                tmp_i_idx = [CLS_token]
                tmp_i_tok = [
                    word_2_idx.get(x, UNK_token) for \
                        x in valid_data[curr_idx][1].split(" ")]
                
                n_token = len(tmp_i_tok)
                if n_token > seq_length:
                    tmp_i_idx += tmp_i_tok[:seq_length]
                    tmp_i_idx += [TRU_token]
                else:
                    tmp_i_idx += tmp_i_tok + [EOS_token]
                
                if test_sample == curr_idx:
                    test_flag = True
                    test_text = " ".join([
                        idx_2_word.get(x) for x in tmp_i_idx])
                
                n_input = len(tmp_i_idx)
                x_valid[n_tmp, :n_input] = tmp_i_idx
            
            # Perform inference. #
            tmp_pred_labels = bert_model.infer(x_valid)
            pred_labels.append(tmp_pred_labels.numpy())
            
            if test_flag:
                batch_idx  = test_sample - id_st
                test_label = tmp_pred_labels[batch_idx]
                
                test_flag  = False
                test_label = test_label.numpy()
                true_label = y_valid[test_sample]
            del x_valid
        
        # Concatenate the predicted labels. #
        pred_labels = np.concatenate(
            tuple(pred_labels), axis=0)
        
        # Compute the accuracy. #
        accuracy = np.sum(np.where(
            pred_labels == y_valid, 1, 0)) / num_valid
        
        # Save the classification report. #
        pred_report = classification_report(
            y_valid, pred_labels, 
            zero_division=0, target_names=target_labels)
        del pred_labels
        
        tmp_report = "bert_emotion_report_mlm.txt"
        with open(tmp_report, "w") as tmp_write:
            tmp_write.write(pred_report)
        
        avg_loss = tot_loss / display_step
        tot_loss = 0.0
        elapsed_tm = (time.time() - start_tm) / 60
        
        print("Iteration:", str(n_iter) + ".")
        print("Elapsed Time:", 
              str(round(elapsed_tm, 2)), "mins.")
        print("Learn Rate:", str(learn_rate_val) + ".")
        print("Average Train Loss:", str(avg_loss) + ".")
        print("Validation Accuracy:", 
              str(round(accuracy * 100, 2)) + "%.")
        
        print("")
        print("Test Text:", test_text)
        print("Pred Label:", idx_2_labels[test_label])
        print("True Label:", idx_2_labels[true_label])
        
        start_tm = time.time()
        train_loss_list.append((n_iter, avg_loss, accuracy))
        
        if n_iter % cooling_step != 0:
            print("-" * 50)
    
    if n_iter % cooling_step == 0:
        # Save the model. #
        save_path = manager.save()
        print("Saved model to {}".format(save_path))
        
        df_col_names  = ["iter", "xent_loss", "val_acc"]
        train_loss_df = pd.DataFrame(
            train_loss_list, columns=df_col_names)
        train_loss_df.to_csv(train_loss_file, index=False)
        del train_loss_df
        
        print("Cooling GPU for 2 minutes.")
        time.sleep(120)
        print("-" * 50)

# Get the performance on the full validation data. #
num_test = len(tuple_valid)
if num_test <= batch_test:
    n_test_batches = 1
elif num_test % batch_test == 0:
    n_test_batches = int(num_test / batch_test)
else:
    n_test_batches = int(num_test / batch_test) + 1

pred_labels = []
for n_val_batch in range(n_test_batches):
    id_st = n_val_batch * batch_test
    if n_val_batch == (n_test_batches-1):
        id_en = num_test
    else:
        id_en = (n_val_batch+1) * batch_test
    curr_batch = id_en - id_st
    
    x_valid = np.zeros(
        [curr_batch, seq_length+2], dtype=np.int32)
    
    x_valid[:, :] = PAD_token
    for n_tmp in range(curr_batch):
        curr_idx = id_st + n_tmp
        
        tmp_i_idx = [CLS_token]
        tmp_i_tok = [
            word_2_idx.get(x, UNK_token) for \
                x in tuple_valid[curr_idx][1].split(" ")]
        
        n_token = len(tmp_i_tok)
        if n_token > seq_length:
            tmp_i_idx += tmp_i_tok[:seq_length]
            tmp_i_idx += [TRU_token]
        else:
            tmp_i_idx += tmp_i_tok + [EOS_token]
            
        n_input = len(tmp_i_idx)
        x_valid[n_tmp, :n_input] = tmp_i_idx
    
    # Perform inference. #
    tmp_pred_labels = bert_model.infer(x_valid)
    pred_labels.append(tmp_pred_labels.numpy())
    del x_valid

y_test = [
    labels_2_idx[x] for x, y in tuple_valid]
pred_labels = np.concatenate(
    tuple(pred_labels), axis=0)
pred_report = classification_report(
    y_test, pred_labels, 
    zero_division=0, target_names=target_labels)

tmp_report = "bert_emotion_report_mlm.txt"
with open(tmp_report, "w") as tmp_write:
    tmp_write.write(pred_report)
print(pred_report)


