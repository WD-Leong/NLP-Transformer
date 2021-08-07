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
| Training Loss | Validation Accuracy |
| ------------- | ------------------- |
| ![training_loss](training_loss.jpg) | ![validation_accuracy](validation_accuracy.jpg) |

## Evaluation Metrics
The final evaluation on the validation dataset is as follows:
```
              precision    recall  f1-score   support

           0       0.92      0.89      0.91     14320
           1       0.85      0.89      0.87     11949
           2       0.91      0.95      0.93     35285
           3       0.87      0.73      0.79      8724
           4       0.93      0.94      0.94     30161
           5       0.83      0.71      0.77      3764

    accuracy                           0.90    104203
   macro avg       0.89      0.85      0.87    104203
weighted avg       0.90      0.90      0.90    104203
```

