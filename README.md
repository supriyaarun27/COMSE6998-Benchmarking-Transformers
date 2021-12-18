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
