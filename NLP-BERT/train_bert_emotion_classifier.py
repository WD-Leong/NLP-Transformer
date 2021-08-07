import time
import numpy as np
import pandas as pd
import pickle as pkl
from collections import Counter
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_bert_module as bert
from sklearn.metrics import classification_report
from nltk.tokenize import wordpunct_tokenize as word_tokenizer

def sub_batch_train_step(
    model, sub_batch_sz, 
    x_encode, x_output, optimizer, 
    learning_rate=1.0e-3, gradient_clip=1.0):
    batch_size = x_encode.shape[0]
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
        
        tmp_encode = x_encode[id_st:id_en, :]
        tmp_output = x_output[id_st:id_en]
        
        with tf.GradientTape() as grad_tape:
            output_tuple  = model(tmp_encode, training=True)
            output_logits = output_tuple[0]
            
            tmp_losses = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_output, logits=output_logits))
        
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

tmp_path = "C:/Users/admin/Desktop/Data/emotion_dataset/"
with open(tmp_path + "merged_training.pkl", "rb") as tmp_file:
    full_data = pkl.load(tmp_file)
n_train = int(0.75*len(full_data))

train_data = full_data[:n_train]
valid_data = full_data[n_train:]

uniq_labels = list(
    sorted(list(pd.unique(train_data["emotions"]))))
labels_dict = dict([
    (uniq_labels[x], x) for x in range(len(uniq_labels))])
idx_2_label = dict([
    (x, uniq_labels[x]) for x in range(len(uniq_labels))])

# Form the vocabulary. #
valid_corpus = list(valid_data["text"].values)
train_corpus = list(train_data["text"].values)

w_counter  = Counter()
max_length = 180
min_count  = 10
tmp_count  = 0
for tmp_text in train_corpus:
    tmp_text = tmp_text.replace("\n", " \n ").strip()
    
    tmp_tokens = [
        x for x in word_tokenizer(
            tmp_text.lower()) if x != ""]
    w_counter.update(tmp_tokens)
    
    tmp_count += 1
    if tmp_count % 25000 == 0:
        proportion = tmp_count / len(train_corpus)
        print(str(round(proportion*100, 2)) + "%", 
              "comments processed.")

vocab_list = sorted(
    [x for x, y in w_counter.items() if y >= min_count])
vocab_list = sorted(
    vocab_list + ["<cls>", "<eos>", "<pad>", "<unk>"])
vocab_size = len(vocab_list)

idx2word = dict([
    (x, vocab_list[x]) for x in range(len(vocab_list))])
word2idx = dict([
    (vocab_list[x], x) for x in range(len(vocab_list))])
print("Vocabulary Size:", str(len(vocab_list)))

# Save the dictionary for inference. #
with open("bert_dict.pkl", "wb") as tmp_file_write:
    pkl.dump(word2idx, tmp_file_write)
    pkl.dump(idx_2_label, tmp_file_write)

# Parameters. #
p_noise = 0.10
grad_clip = 1.00

steps_max  = 10000
n_classes  = len(labels_dict)
batch_size = 256
sub_batch  = 64
batch_test = 128

hidden_size  = 256
warmup_steps = 5000
cooling_step = 500
display_step = 250
restore_flag = False

model_ckpt_dir  = "Model/bert_emotion_model"
train_loss_file = "bert_emotion_losses.csv"

# Define the classifier model. #
bert_model = bert.BERT_Classifier(
    n_classes, 3, 4, hidden_size, 
    4*hidden_size, word2idx, max_length)
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
    labels_dict[x] for x in valid_data["emotions"].values]
y_train = [
    labels_dict[x] for x in train_data["emotions"].values]

cls_token = word2idx["<cls>"]
pad_token = word2idx["<pad>"]
unk_token = word2idx["<unk>"]
eos_token = word2idx["<eos>"]

x_train = []
for n in range(len(train_corpus)):
    tmp_text = train_corpus[n].replace(
        "\n", " \n ").strip()
    
    tmp_tokens = [
        x for x in word_tokenizer(
            tmp_text.lower()) if x != ""]
    
    m = len(tmp_tokens)
    x_train.append([word2idx.get(
        x, unk_token) for x in tmp_tokens])
del train_corpus

x_valid = pad_token * np.ones(
    [len(valid_data), max_length], dtype=np.int32)

x_valid[:, 0] = cls_token
for n in range(len(valid_data)):
    tmp_text = valid_corpus[n].replace(
        "\n", " \n ").strip()
    
    tmp_tokens = [
        x for x in word_tokenizer(
            tmp_text.lower()) if x != ""]
    
    m = len(tmp_tokens) + 1
    x_valid[n, 1:m] = [
        word2idx.get(x, unk_token) for x in tmp_tokens]
    x_valid[n, m]  = eos_token
del valid_corpus

num_valid = len(valid_data)
if num_valid <= batch_size:
    n_val_batches = 1
elif num_valid % batch_size == 0:
    n_val_batches = int(num_valid / batch_size)
else:
    n_val_batches = int(num_valid / batch_size) + 1

# Train the model. #
print("Training BERT Model.")
print(str(len(train_data)), "training data.")
print(str(num_valid), "validation data.")

tot_loss = 0.0
num_data = len(train_data)

n_iter = ckpt.step.numpy().astype(np.int32)
sentence_tok = np.zeros(
    [batch_size, max_length], dtype=np.int32)
labels_array = \
    np.zeros([batch_size], dtype=np.int32)

start_tm = time.time()
while n_iter < steps_max:
    # Constant warmup rate. #
    step_val = float(max(n_iter+1, warmup_steps))**(-0.5)
    learn_rate_val = float(hidden_size)**(-0.5) * step_val
    
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    labels_array[:] = n_classes
    sentence_tok[:, :] = pad_token
    sentence_tok[:, 0] = cls_token
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_label = y_train[tmp_index]
        tmp_i_tok = x_train[tmp_index]
        
        # Generate noise. #
        n_input  = len(tmp_i_tok)
        n_decode = n_input + 1
        in_noisy = [unk_token] * n_input
        tmp_flag = np.random.binomial(1, p_noise, size=n_input)
        in_noise = [
            tmp_i_tok[x] if tmp_flag[x] == 1 else \
                in_noisy[x] for x in range(n_input)]
        
        labels_array[n_index] = tmp_label
        if p_noise > 0.0:
            sentence_tok[n_index, 1:n_decode] = in_noise
        else:
            sentence_tok[n_index, 1:n_decode] = tmp_i_tok
        
        sentence_tok[n_index, n_decode] = eos_token
        del tmp_i_tok, tmp_label, tmp_index
        del tmp_flag, n_input, n_decode, in_noisy, in_noise
    
    tmp_loss = sub_batch_train_step(
        bert_model, sub_batch, 
        sentence_tok, labels_array, bert_optim, 
        learning_rate=learn_rate_val, gradient_clip=grad_clip)
    
    # Increment the step. #
    n_iter += 1
    ckpt.step.assign_add(1)
    
    tot_loss += tmp_loss.numpy()
    if n_iter % display_step == 0:
        # Get the validation accuracy. #
        pred_labels = []
        for n_val_batch in range(n_val_batches):
            id_st = n_val_batch * batch_size
            if n_val_batch == (n_val_batches-1):
                id_en = num_valid
            else:
                id_en = (n_val_batch+1) * batch_size
            
            # Perform inference. #
            tmp_pred_labels = bert_model.infer(
                x_valid[id_st:id_en, :]).numpy()
            pred_labels.append(tmp_pred_labels)
        
        # Concatenate the predicted labels. #
        pred_labels = np.concatenate(
            tuple(pred_labels), axis=0)
        
        # Compute the accuracy. #
        accuracy = np.sum(np.where(
            pred_labels == y_valid, 1, 0)) / num_valid
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
pred_labels = []
for n_val_batch in range(n_val_batches):
    id_st = n_val_batch * batch_size
    if n_val_batch == (n_val_batches-1):
        id_en = num_valid
    else:
        id_en = (n_val_batch+1) * batch_size
    
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

# Extract the training losses. #
df_col_names  = ["iter", "xent_loss", "val_acc"]
train_loss_df = pd.DataFrame(
    train_loss_list, columns=df_col_names)

# Plot the training loss. #
fig, ax = plt.subplots()
ax.plot(train_loss_df["iter"], train_loss_df["xent_loss"])
ax.set(xlabel="Step", ylabel="Training Loss")
fig.suptitle("Emotion Classification Training Loss")
fig.savefig("training_loss.jpg", dpi=199)
plt.close("all")
del fig, ax
