# Bidirectional Encoder Representations from Transformers
This repository contains an implementation of [Bidirectional Encoder Representations from Transformers](https://arxiv.org/abs/1810.04805v2). The model is trained on a an [emotional classification dataset](https://github.com/dair-ai/emotion_dataset). To train the model, run
```
python train_bert_emotion_classifier.py
```
and run
```
python infer_bert_emotion_classifier.py
```
to perform inference. Please note that this implementation trains the model from scratch using the labelled data and no fine-tuning or transfer learning is performed.

## Training and Validation Progress
The plots below show the training and validation progress of the model.

