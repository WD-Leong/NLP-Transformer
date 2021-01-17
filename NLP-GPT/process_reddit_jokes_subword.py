
import time
import json
import pickle as pkl
from collections import Counter
import byte_pair_encoding as bpe
from nltk.tokenize import wordpunct_tokenize as word_tokenizer

print("Loading the data.")
start_tm = time.time()

tmp_file = "C:/Users/admin/Desktop/Codes/reddit_jokes.json"
tmp_data = json.loads(open(tmp_file).read())
max_len  = 50

# Extract the data. #
tmp_jokes_tuple = []
for tmp_row in tmp_data:
    if tmp_row["body"].find(tmp_row["title"]) != -1:
        tmp_joke = tmp_row["body"]
    elif tmp_row["title"].find(tmp_row["body"]) != -1:
        tmp_joke = tmp_row["title"]
    else:
        tmp_joke = tmp_row["title"] + " " + tmp_row["body"]
    
    if tmp_row["score"] >= 5:
        if tmp_row["score"] < 12:
            tmp_class = "bad_joke"
        elif tmp_row["score"] < 50:
            tmp_class = "ok_joke"
        else:
            tmp_class = "good_joke"
        
        tmp_jokes_tuple.append((tmp_class, tmp_joke))
del tmp_row, tmp_joke
print("Total of", str(len(tmp_jokes_tuple)), "jokes loaded.")

# Process the data. #
tmp_jokes_filtered = []

w_counter = Counter()
for tmp_class, tmp_joke in tmp_jokes_tuple:
    tmp_joke = \
        tmp_joke.replace("\n", " \n ").replace("\'", " ")
    tmp_tokens = [
        x for x in word_tokenizer(tmp_joke.lower()) if x != ""]
    
    if len(tmp_tokens) <= max_len:
        w_counter.update(tmp_tokens)
        tmp_jokes_filtered.append((tmp_class, tmp_joke))
    del tmp_tokens

print("Total of", str(len(tmp_jokes_filtered)), "jokes filtered.")
del tmp_jokes_tuple

word_counts = []
for word, count in w_counter.items():
    tmp_word = "<" + word + ">"
    tmp_word = "".join([x+" " for x in tmp_word]).strip()
    word_counts.append((tmp_word, count))
word_counts = dict(word_counts)

elapsed_tm = (time.time() - start_tm) / 60
print("Total of", str(len(word_counts)), "words.")
print("Elapsed Time:", str(elapsed_tm), "mins.")

# Fit the subword vocabulary. #
print("Fitting subword vocabulary.")
start_tm = time.time()

n_iters = 1000
vocab_size = 8000
tuple_out  = bpe.learn_subword_vocab(
    word_counts, n_iters, vocab_size=vocab_size)

subword_vocab_list = tuple_out[0]
idx2subword = tuple_out[1]
subword2idx = tuple_out[2]

# Add the conditional classes for joke generation. #
tmp_classes = ["<bad_joke>", "<ok_joke>", "<good_joke>"]
for tmp_class in tmp_classes:
    curr_vocab_size = len(subword_vocab_list)
    
    subword_vocab_list.append(tmp_class)
    idx2subword[curr_vocab_size] = tmp_class
    subword2idx[tmp_class] = curr_vocab_size

elapsed_tm = (time.time() - start_tm) / 60
print("Total Sub-word Vocabulary size:", 
      str(len(subword_vocab_list)), "sub-word tokens")
print("Elapsed Time:", str(elapsed_tm), "mins.")

# Encode the corpus to subword tokens. #
print("Encoding the corpus to subwords.")
start_tm = time.time()

new_jokes_tuple = []
for tmp_class, tmp_joke in tmp_jokes_filtered:
    tmp_joke_cls = tmp_class + " " + tmp_joke
    tmp_joke_sw = bpe.bp_encode(
        tmp_joke_cls, subword_vocab_list, subword2idx)
    new_jokes_tuple.append(tmp_joke_sw)

elapsed_tm = (time.time() - start_tm) / 60
print("Elapsed Time:", str(elapsed_tm), "mins.")

# Save the data. #
print("Saving the file.")

tmp_pkl_file = \
    "C:/Users/admin/Desktop/Codes/reddit_jokes_subword.pkl"
with open(tmp_pkl_file, "wb") as tmp_file_save:
    pkl.dump(new_jokes_tuple, tmp_file_save)
    pkl.dump(idx2subword, tmp_file_save)
    pkl.dump(subword2idx, tmp_file_save)
