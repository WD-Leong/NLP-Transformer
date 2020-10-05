# NLP Generative Pre-Training (GPT) Model
This repository includes the codes that I have modified for the GPT model, following the publication of [Open AI's GPT Model](https://openai.com/blog/better-language-models/). In particular, the changes include (i) the addition of a learnt positional embedding vector in each layer and (ii) the addition of a residual connection between the input embedding and the output embedding layer.

This repository includes a code to train the data on the [Movie Dialogue](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) dataset, where the preparation of the data follows this [script](https://github.com/suriyadeepan/datasets/blob/master/seq2seq/cornell_movie_corpus/scripts/prepare_data.py) closely. 
