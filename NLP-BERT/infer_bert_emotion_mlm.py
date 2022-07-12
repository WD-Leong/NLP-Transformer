import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_bert_keras as bert

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

CLS_token = word_2_idx["CLS"]
EOS_token = word_2_idx["EOS"]
PAD_token = word_2_idx["PAD"]
UNK_token = word_2_idx["UNK"]
MSK_token = word_2_idx["MSK"]
TRU_token = word_2_idx["TRU"]

# Parameters. #
p_keep = 0.90
p_mask = 0.15

num_layers = 3
num_heads  = 4
n_classes  = len(labels_2_idx)

seq_length  = 30
hidden_size = 256
ffwd_size   = 4*hidden_size

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
    [1, seq_length+2], dtype=np.int32)
tmp_pred = bert_model(tmp_zero, training=True)[0]

print(bert_model.summary())
del tmp_zero, tmp_pred

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Model restored from {}".format(
        manager.latest_checkpoint))
else:
    print("Error: No latest checkpoint found.")
n_iter = ckpt.step.numpy().astype(np.int32)

# Train the model. #
print("BERT Model Inference.")
print("(" + str(n_iter), "iterations.)")
print("-" * 50)

while True:
    tmp_phrase = input("Enter text: ")
    tmp_phrase = tmp_phrase.lower().strip()
    
    if tmp_phrase == "":
        break
    else:
        tmp_test_tokens = np.zeros(
            [1, seq_length+2], dtype=np.int32)
        tmp_test_tokens[:, :] = PAD_token
        
        tmp_i_idx = [CLS_token]
        tmp_i_tok = [word_2_idx.get(
            x, UNK_token) for x in tmp_phrase.split(" ")]
        
        n_token = len(tmp_i_tok)
        if n_token > seq_length:
            tmp_i_idx += tmp_i_tok[:seq_length]
            tmp_i_idx += [TRU_token]
        else:
            tmp_i_idx += tmp_i_tok + [EOS_token]
        
        n_input = len(tmp_i_idx)
        tmp_test_tokens[0, :n_input] = tmp_i_idx
        
        # Perform inference. #
        tmp_pred_labels = bert_model.infer(
            tmp_test_tokens).numpy()[0]
        
        print("Input Text:", tmp_phrase)
        print("Pred Label:", idx_2_labels[tmp_pred_labels])
        print("-" * 50)
        
        del n_token, n_input, tmp_pred_labels
        del tmp_test_tokens, tmp_i_idx, tmp_i_tok


