
import re
import pickle as pkl
from collections import Counter

tmp_path = "C:/Users/admin/Desktop/Data/twitter_en/"

tmp_line_file = tmp_path + "twitter_en.txt"
with open(tmp_line_file, "r", encoding='utf-8', errors='ignore') as tmp_file:
    tmp_lines = tmp_file.readlines()

q_len = 15
a_len = 15

q_counter = Counter()
a_counter = Counter()
tmp_tuple = []

for n in range(0, len(tmp_lines), 2):
    tmp_q = tmp_lines[n]
    tmp_a = tmp_lines[n+1]
    
    tmp_q = re.sub(r"[^\w\s]", " ", tmp_q)
    tmp_a = re.sub(r"[^\w\s]", " ", tmp_a)
    tmp_q = [x for x in tmp_q.split(" ") if x != ""]
    tmp_a = [x for x in tmp_a.split(" ") if x != ""]
    
    if len(tmp_q) == 0 or len(tmp_a) == 0:
        continue
    elif len(tmp_q) <= q_len and len(tmp_a) <= a_len:
        q_counter.update(tmp_q)
        a_counter.update(tmp_a)
        tmp_tuple.append((" ".join(tmp_q), " ".join(tmp_a)))
print(len(tmp_tuple), "pairs found.")

# Generte the vocabulary. #
vocab_size = 20000
q_vocab_list = sorted([
    x for x, y in q_counter.most_common(vocab_size)])
q_vocab_list = ["SOS", "PAD", "UNK"] + q_vocab_list
a_vocab_list = sorted([
    x for x, y in a_counter.most_common(vocab_size)])
a_vocab_list = ["SOS", "EOS", "PAD", "UNK"] + a_vocab_list

q_idx2word = dict([
    (x, q_vocab_list[x]) for x in range(len(q_vocab_list))])
q_word2idx = dict([
    (q_vocab_list[x], x) for x in range(len(q_vocab_list))])
a_idx2word = dict([
    (x, a_vocab_list[x]) for x in range(len(a_vocab_list))])
a_word2idx = dict([
    (a_vocab_list[x], x) for x in range(len(a_vocab_list))])

# Save the file. #
tmp_pkl_file = tmp_path + "twitter_corpus.pkl"
with open(tmp_pkl_file, "wb") as tmp_file_save:
    pkl.dump(tmp_tuple, tmp_file_save)
    pkl.dump(q_idx2word, tmp_file_save)
    pkl.dump(q_word2idx, tmp_file_save)
    pkl.dump(a_idx2word, tmp_file_save)
    pkl.dump(a_word2idx, tmp_file_save)
