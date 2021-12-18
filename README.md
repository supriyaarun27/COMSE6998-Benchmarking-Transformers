# Benchmarking-Transformers

## Final Project for COMSE6998 Practical Deep Learning System Performance Fall 2021

### Contributers :
#### Supriya Arun (sa3982) 
#### Arvind Kanesan Rathna (ak4728)


## Summary 

Comparison, benchmarking and analysis of performance metrics for three transformer models BERT, DistilBert, and SqueezeBERT. We evaluate tradeoffs in accuracy, computational requirements, and dollar cost on Question and Answering (SQuAD benchmark). 

## Running the Code 

### Step 1 : Hugging Face Setup 

Install the huggingface_hub library with pip in your environment:

```

python -m pip install huggingface_hub

```

Once you have successfully installed the huggingface_hub library, log in to your Hugging Face account:

```
huggingface-cli login
```
Login with the token you can get on your hugging face account. 

### Step 2 : Install Dependencies 

Install the required dependencies in the the requirements.txt file 

### Step 3 : Run Training Code 

```
python trianing.py

```

The training code will run with the following default parameters : 

model_checkpoint = "distilbert-base-uncased"
batch_size = 16
epochs = 3

Change these as required. 

Once the training is finished your model will be uploaded to the hugging face Model Hub. 

## Approach and Solution Diagram

![alt text](https://github.com/supriyaarun27/COMSE6998-Benchmarking-Transformers/blob/main/assets/flowchart.png?raw=true)


## Our Results :

DistilBERT and SqueezeBERT both compromise F1 and EM scores to a small extent for faster inference speed. 

EM (Exact Match) : A binary measure of whether the system output matches the ground truth answer exactly

![alt text](https://github.com/supriyaarun27/COMSE6998-Benchmarking-Transformers/blob/main/assets/cost.png?raw=true)

Architecture changes that we think explain this behavior : 

DistilBERT 
The model architecture is similar to BERT but has half the number of layers of BERT. Using knowledge distillation the model is able to retain 97% of BERT's language capabilities. 
[ Our results (F1 score ratio) of 96.8% correlates with the paper's claims ]

SqueezeBERT
This model is built for edge devices and has BERT's fully connected layers replaced with "grouped convolutions" this makes the model 4x faster but leads to some loss in accuracy.  

## Cost for training on P100-GCP

DistilBERT is most economical to finetune
Anamoly: SqueezeBERT is most expensive despite being a smaller model than BERT. Requires further investigation.

![alt text](https://github.com/supriyaarun27/COMSE6998-Benchmarking-Transformers/blob/main/assets/metrics.png?raw=true)

