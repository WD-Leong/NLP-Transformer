
import re
import pickle as pkl
from collections import Counter

print("Loading the data.")
tmp_path = "C:/Users/admin/Desktop/Codes/"

tmp_line_file = tmp_path + "movie_lines.txt"
with open(tmp_line_file, "r", encoding='utf-8', errors='ignore') as tmp_file:
    tmp_lines = tmp_file.readlines()

tmp_conv_file = tmp_path + "movie_conversations.txt"
with open(tmp_conv_file, "r", encoding='utf-8', errors='ignore') as tmp_file:
    tmp_convs = tmp_file.readlines()

id2line = {}
for tmp_line in tmp_lines:
    tmp_split = str(tmp_line).split(" +++$+++ ")
    if len(tmp_split) == 5:
        id2line[tmp_split[0]] = tmp_split[4]

convs = []
for tmp_conv in tmp_convs[:-1]:
    tmp_split = str(tmp_conv).replace("\\n", "").split(
        " +++$+++ ")[-1][1:-1].replace("'","").replace(" ","")
    tmp_split = tmp_split.replace("]", "")
    
    tmp_ids = [str(x.encode("utf-8")).replace(
        "b'", "").replace("'", "") for x in tmp_split.split(",")]
    convs.append(tmp_ids)

q_len = 20
a_len = 20

w_counter = Counter()
tmp_tuple = []
for conv in convs:
    for i in range(len(conv)-1):
        tmp_qns = id2line[conv[i]].lower().replace(
            "\\u", " ").replace("\\i", " ").replace("\n", " ").replace("\t", " ")
        tmp_qns = re.sub(r"[^\w\s]", " ", tmp_qns)
        tmp_qns = [x for x in tmp_qns.split(" ") if x != ""]

        tmp_ans = id2line[conv[i+1]].lower().replace(
            "\\u", " ").replace("\\i", " ").replace("\n", " ").replace("\t", " ")
        tmp_ans = re.sub(r"[^\w\s]", " ", tmp_ans)
        tmp_ans = [x for x in tmp_ans.split(" ") if x != ""]

        if len(tmp_qns) == 0 or len(tmp_ans) == 0:
            continue
        elif len(tmp_qns) <= q_len and len(tmp_ans) <= a_len:
            w_counter.update(tmp_qns)
            w_counter.update(tmp_ans)
            tmp_tuple.append((" ".join(tmp_qns), " ".join(tmp_ans)))

vocab_size = 20000
vocab_list = sorted([x for x, y in w_counter.most_common(vocab_size)])
vocab_list = ["SOS", "EOS", "PAD", "UNK"] + vocab_list

idx2word = dict([
    (x, vocab_list[x]) for x in range(len(vocab_list))])
word2idx = dict([
    (vocab_list[x], x) for x in range(len(vocab_list))])

tmp_pkl_file = tmp_path + "movie_dialogues_20.pkl"
with open(tmp_pkl_file, "wb") as tmp_file_save:
    pkl.dump(tmp_tuple, tmp_file_save)
    pkl.dump(idx2word, tmp_file_save)
    pkl.dump(word2idx, tmp_file_save)
