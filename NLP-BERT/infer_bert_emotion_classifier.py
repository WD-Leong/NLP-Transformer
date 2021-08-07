import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_bert_module as bert
from nltk.tokenize import wordpunct_tokenize as word_tokenizer

# Load the dictionary for inference. #
print("Loading and processing the data.")

with open("bert_dict.pkl", "rb") as tmp_file_load:
    word2idx = pkl.load(tmp_file_load)
    idx_2_label = pkl.load(tmp_file_load)

cls_token = word2idx["<cls>"]
pad_token = word2idx["<pad>"]
unk_token = word2idx["<unk>"]
eos_token = word2idx["<eos>"]
print("Vocabulary Size:", str(len(word2idx)))

# Parameters. #
max_length = 180
n_classes  = len(idx_2_label)
hidden_size  = 256

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

# Format the data before training. #
print("Infering emotions using BERT model", 
      "(" + str(int(ckpt.step)) + " iterations).")
print("-" * 50)

sentence_tok = np.zeros(
    [1, max_length], dtype=np.int32)
while True:
    tmp_input = input("Enter input: ")
    if tmp_input.lower().strip() == "":
        break
    else:
        tmp_input  = tmp_input.lower().strip()
        tmp_tokens = [
            x for x in word_tokenizer(tmp_input) if x != ""]
        tmp_i_idx  = [
            word2idx.get(x, unk_token) for x in tmp_tokens]
        
        n_tokens = len(tmp_i_idx)
        n_decode = n_tokens + 1
        
        sentence_tok[:, :] = pad_token
        sentence_tok[:, 0] = cls_token
        sentence_tok[:, 1:n_decode] = tmp_i_idx
        sentence_tok[:, n_decode] = eos_token
        
        pred_label = bert_model.infer(
            sentence_tok)[0].numpy()
        
        print("Input Phrase:")
        print(tmp_input)
        print("Predicted Emotion:", str(idx_2_label[pred_label]))
        print("-" * 50)
