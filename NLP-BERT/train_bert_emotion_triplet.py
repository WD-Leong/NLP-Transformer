import time
import numpy as np
import pandas as pd
import pickle as pkl
from collections import Counter
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_bert_cls_keras as bert
from nltk.tokenize import wordpunct_tokenize
from sklearn.metrics import classification_report

def sub_batch_train_step(
    model, sub_batch_sz, 
    x_anc, x_pos, x_neg, x_label, 
    optimizer, alpha=5.0, reg_triplet=0.10, 
    learning_rate=1.0e-3, gradient_clip=1.0):
    batch_size = x_anc.shape[0]
    
    if batch_size <= sub_batch_sz:
        sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        sub_batch = int(batch_size / sub_batch_sz)
    else:
        sub_batch = int(batch_size / sub_batch_sz) + 1
    optimizer.lr.assign(learning_rate)
    
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
        
        tmp_out = x_label[id_st:id_en]
        tmp_anc = x_anc[id_st:id_en, :]
        tmp_pos = x_pos[id_st:id_en, :]
        tmp_neg = x_neg[id_st:id_en, :]
        
        with tf.GradientTape() as grad_tape:
            anc_outputs = model(
                tmp_anc, training=True)
            anc_logits  = anc_outputs[0]
            
            embed_anc = model(
                tmp_anc, training=True)[1][:, 0, :]
            embed_pos = model(
                tmp_pos, training=True)[1][:, 0, :]
            embed_neg = model(
                tmp_neg, training=True)[1][:, 0, :]
            
            # Triplet loss. #
            tmp_pos_dist = tf.reduce_mean(tf.square(
                embed_anc - embed_pos), axis=1)
            tmp_neg_dist = tf.reduce_mean(tf.square(
                embed_anc - embed_neg), axis=1)
            triplet_loss = tf.reduce_sum(tf.maximum(
                0.0, tmp_pos_dist - tmp_neg_dist + alpha))
            
            # Classification loss. #
            cls_losses = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_out, logits=anc_logits))
            tmp_losses = tf.add(
                cls_losses, reg_triplet*triplet_loss)
        
        # Accumulate the gradients. #
        tot_losses += tmp_losses
        
        tmp_gradients = \
            grad_tape.gradient(tmp_losses, model_params)
        acc_gradients = [
            (acc_grad + grad) for acc_grad, grad \
                in zip(acc_gradients, tmp_gradients)]
    
    # Update using the optimizer. #
    avg_losses = tot_losses / batch_size
    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    
    clipped_gradients, _ = tf.clip_by_global_norm(
        acc_gradients, gradient_clip)
    optimizer.apply_gradients(
        zip(clipped_gradients, model_params))
    return avg_losses

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

# Parameters. #
p_noise = 0.10
grad_clip = 1.00

steps_max  = 10000
n_classes  = len(labels_2_idx)
batch_size = 256
sub_batch  = 32
batch_test = 64
max_length = 180

hidden_size  = 256
warmup_steps = 5000
cooling_step = 100
display_step = 50
restore_flag = False

model_ckpt_dir  = "Model/bert_emotion_model"
train_loss_file = "bert_emotion_losses.csv"

# Define the classifier model. #
bert_model = bert.BERTClassifier(
    n_classes, 3, 4, hidden_size, 
    4*hidden_size, vocab_size, max_length)
bert_optim = tfa.optimizers.AdamW(
    beta_1=0.9, beta_2=0.98, 
    epsilon=1.0e-9, weight_decay=1.0e-4)

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    bert_model=bert_model, 
    bert_optim=bert_optim)

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

# Format the data before training. #
print("Formatting training and validation data.")

y_valid = [
    labels_2_idx[x[0]] for x in valid_data]
y_train = [
    labels_2_idx[x[0]] for x in train_data]

cls_token = word_2_idx["CLS"]
pad_token = word_2_idx["PAD"]
unk_token = word_2_idx["UNK"]
eos_token = word_2_idx["EOS"]

x_valid = pad_token * np.ones(
    [len(valid_data), max_length], dtype=np.int32)

x_valid[:, 0] = cls_token
for n in range(len(valid_data)):
    tmp_tokens = [
        word_2_idx.get(x, unk_token) for \
            x in valid_data[n][1].split(" ")]
    
    m = len(tmp_tokens) + 2
    x_valid[n, 1:m] = tmp_tokens + [eos_token]

num_valid = 2500
if num_valid <= batch_test:
    n_val_batches = 1
elif num_valid % batch_test == 0:
    n_val_batches = int(num_valid / batch_test)
else:
    n_val_batches = int(num_valid / batch_test) + 1

# Train the model. #
print("Training BERT Model.")
print(str(len(train_data)), "training data.")
print(str(len(valid_data)), "validation data.")
print("Approximating on", str(num_valid), "data.")
print("-" * 50)

tot_loss = 0.0
num_data = len(train_data)

n_iter = ckpt.step.numpy().astype(np.int32)
labels_array = np.zeros(
    [batch_size], dtype=np.int32)
sentence_anc = np.zeros(
    [batch_size, max_length], dtype=np.int32)
sentence_pos = np.zeros(
    [batch_size, max_length], dtype=np.int32)
sentence_neg = np.zeros(
    [batch_size, max_length], dtype=np.int32)

start_tm = time.time()
while n_iter < steps_max:
    # Constant warmup rate. #
    step_val = float(max(n_iter+1, warmup_steps))**(-0.5)
    learn_rate_val = float(hidden_size)**(-0.5) * step_val
    
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    labels_array[:] = n_classes
    sentence_anc[:, :] = pad_token
    sentence_pos[:, :] = pad_token
    sentence_neg[:, :] = pad_token
    
    sentence_anc[:, 0] = cls_token
    sentence_pos[:, 0] = cls_token
    sentence_neg[:, 0] = cls_token
    
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        neg_index = tmp_index + num_data
        neg_index = neg_index % num_data
        
        tmp_label = y_train[tmp_index]
        tmp_i_pos = [
            word_2_idx.get(x, unk_token) for x in \
                train_data[tmp_index][1].split(" ")]
        tmp_i_neg = [
            word_2_idx.get(x, unk_token) for x in \
                train_data[neg_index][1].split(" ")]
        
        # Generate noise. #
        n_input  = len(tmp_i_pos)
        n_neg_in = len(tmp_i_neg) + 1
        n_decode = n_input + 1
        in_noisy = [unk_token] * n_input
        tmp_flag = np.random.binomial(
            1, p_noise, size=n_input)
        in_noise = [
            tmp_i_pos[x] if tmp_flag[x] == 1 else \
                in_noisy[x] for x in range(n_input)]
        
        labels_array[n_index] = tmp_label
        if p_noise > 0.0:
            sentence_anc[n_index, 1:n_decode] = in_noise
        else:
            sentence_anc[n_index, 1:n_decode] = tmp_i_pos
        
        sentence_pos[n_index, 1:n_decode] = tmp_i_pos
        sentence_neg[n_index, 1:n_neg_in] = tmp_i_neg
        
        sentence_anc[n_index, n_decode] = eos_token
        sentence_pos[n_index, n_decode] = eos_token
        sentence_neg[n_index, n_neg_in] = eos_token
        
        del tmp_i_pos, tmp_i_neg, tmp_label
        del n_neg_in, tmp_index, neg_index
        del tmp_flag, n_input, n_decode, in_noisy, in_noise
    
    tmp_loss = sub_batch_train_step(
        bert_model, sub_batch, sentence_anc, 
        sentence_pos, sentence_neg, labels_array, bert_optim, 
        learning_rate=learn_rate_val, gradient_clip=grad_clip)
    
    # Increment the step. #
    n_iter += 1
    ckpt.step.assign_add(1)
    
    tot_loss += tmp_loss.numpy()
    if n_iter % display_step == 0:
        # Get the validation accuracy. #
        pred_labels = []
        for n_val_batch in range(n_val_batches):
            id_st = n_val_batch * batch_test
            if n_val_batch == (n_val_batches-1):
                id_en = num_valid
            else:
                id_en = (n_val_batch+1) * batch_test
            
            # Perform inference. #
            tmp_pred_labels = bert_model.infer(
                x_valid[id_st:id_en, :]).numpy()
            pred_labels.append(tmp_pred_labels)
        
        # Concatenate the predicted labels. #
        pred_labels = np.concatenate(
            tuple(pred_labels), axis=0)
        
        # Compute the accuracy. #
        y_approx = y_valid[:num_valid]
        accuracy = np.sum(np.where(
            pred_labels == y_approx, 1, 0)) / num_valid
        del pred_labels
        
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

# Get the performance on the validation data. #
num_valid = len(valid_data)
if num_valid <= batch_test:
    n_val_batches = 1
elif num_valid % batch_test == 0:
    n_val_batches = int(num_valid / batch_test)
else:
    n_val_batches = int(num_valid / batch_test) + 1

pred_labels = []
for n_val_batch in range(n_val_batches):
    id_st = n_val_batch * batch_test
    if n_val_batch == (n_val_batches-1):
        id_en = num_valid
    else:
        id_en = (n_val_batch+1) * batch_test
    
    # Perform inference. #
    tmp_pred_labels = bert_model.infer(
        x_valid[id_st:id_en, :]).numpy()
    pred_labels.append(tmp_pred_labels)

pred_labels = np.concatenate(
    tuple(pred_labels), axis=0)
pred_report = classification_report(y_valid, pred_labels)
with open("validation_report.txt", "w") as tmp_write:
    tmp_write.write(pred_report)
print(pred_report)
print("Complete.")
