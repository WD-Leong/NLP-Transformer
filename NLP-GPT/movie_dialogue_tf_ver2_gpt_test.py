
import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tf_ver2_gpt as tf_gpt

# Model Parameters. #
batch_size = 256
sub_batch  = 64
seq_length = 21
num_heads  = 4
num_layers = 3

prob_keep = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size
warmup_flag = True
cooling_step = 1000

tmp_path = "C:/Users/admin/Desktop/Codes/"
model_ckpt_dir  = \
    "C:/Users/admin/Desktop/TF_Models/dialogue_transformer_gpt"
train_loss_file = \
    "C:/Users/admin/Desktop/Codes/train_loss_dialogue_gpt.csv"

# Load the data. #
tmp_pkl_file = \
    "C:/Users/admin/Desktop/Codes/movie_dialogues.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    data_tuple = pkl.load(tmp_load_file)
    idx2word = pkl.load(tmp_load_file)
    word2idx = pkl.load(tmp_load_file)

vocab_size = len(word2idx)
print("Vocabulary Size:", str(vocab_size)+".")

SOS_token = word2idx["SOS"]
EOS_token = word2idx["EOS"]
PAD_token = word2idx["PAD"]
UNK_token = word2idx["UNK"]

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Build the Transformer. #
print("Building the GPT Model.")
start_time = time.time()

gpt_model = tf_gpt.GPT_Network(
    num_layers, num_heads, hidden_size, 
    ffwd_size, vocab_size, seq_length, 
    embed_size=hidden_size, p_keep=prob_keep)
gpt_optimizer = tf.keras.optimizers.Adam(
    beta_1=0.9, beta_2=0.98, epsilon=1.0e-9)

elapsed_time = (time.time()-start_time) / 60
print("GPT Model Built", "("+str(elapsed_time)+" mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    gpt_model=gpt_model, 
    gpt_optimizer=gpt_optimizer)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Model restored from {}".format(manager.latest_checkpoint))
else:
    print("Error: No latest checkpoint found.")

# Train the Transformer model. #
tmp_test_in = np.zeros([1, seq_length], dtype=np.int32)

# Warmup learning schedule. #
n_iter = ckpt.step.numpy().astype(np.int32)

print("-" * 50)
print("Testing the GPT Network", 
      "(" + str(n_iter) + " iterations).")
print("-" * 50)

# Update the neural network's weights. #
while True:
    tmp_in_phrase = input("Enter input phrase: ")
    if tmp_in_phrase == "":
        break
    else:
        tmp_test_in[:, :] = PAD_token
        
        tmp_in_phrase = tmp_in_phrase.lower()
        tmp_p_tokens  = tmp_in_phrase.split(" ")
        
        tmp_p_idx = [word2idx.get(
            x, UNK_token) for x in tmp_p_tokens]
        n_tokens  = len(tmp_p_idx) + 1
        tmp_test_in[0, :n_tokens] = tmp_p_idx + [SOS_token]
        
        tmp_infer = gpt_model.infer(tmp_test_in[:, :n_tokens])
        gen_phrase = [idx2word[x] for x in tmp_infer[0].numpy()]
        gen_output = " ".join(gen_phrase[n_tokens:])
        
        print("")
        print("Input Phrase:")
        print(tmp_in_phrase)
        print("Generated Response:")
        print(gen_output)
        del n_tokens
        print("-" * 50)
    


