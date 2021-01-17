
import time
import numpy as np
import pandas as pd
import pickle as pkl
import byte_pair_encoding as bpe

import tensorflow as tf
import tf_ver2_gpt as tf_transformer


# Model Parameters. #
batch_size = 256
sub_batch  = 64
seq_length = 31
num_heads  = 4
num_layers = 3

prob_keep = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size
warmup_flag = True
cooling_step = 2000

model_ckpt_dir = \
    "C:/Users/admin/Desktop/TF_Models/gpt_ver2_subword_reddit"

# Load the data. #
tmp_pkl_file = \
    "C:/Users/admin/Desktop/Codes/reddit_jokes_subword.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    full_data = pkl.load(tmp_load_file)
    idx2subword = pkl.load(tmp_load_file)
    subword2idx = pkl.load(tmp_load_file)
subword_vocab = [x for x, y in list(subword2idx.items())]

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

# Build the Transformer. #
print("Building the GPT Model.")
start_time = time.time()

gpt_model = tf_transformer.GPT_Network(
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
print(str(num_data), "training samples.")
print("-" * 50)

# Update the neural network's weights. #
while True:
    tmp_phrase = input("Enter input seed: ")
    if tmp_phrase == "":
        break
    else:
        tmp_test_in[:, :] = PAD_token
        
        
        tmp_phrase  = tmp_phrase.lower()
        tmp_p_index = bpe.bp_encode(
            tmp_phrase, subword_vocab, subword2idx)
        
        n_tokens = len(tmp_p_index)
        tmp_test_in[0, :n_tokens] = tmp_p_index
        
        tmp_infer  = gpt_model.infer(tmp_test_in[:, :n_tokens])
        gen_phrase = bpe.bp_decode(
            tmp_infer[0].numpy(), idx2subword)
        gen_phrase = \
            " ".join(gen_phrase).replace("<", "").replace(">", "")
        
        print("Input Phrase:")
        print(tmp_phrase)
        print("Generated Phrase:")
        print(gen_phrase)
        print("-" * 50)

