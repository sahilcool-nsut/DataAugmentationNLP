#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import clear_output


# In[2]:


from IPython.display import clear_output 
get_ipython().system('python3 --version')


# In[4]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install transformers')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install torch')
get_ipython().system('pip install pyarrow')
get_ipython().system('pip install datasets')
get_ipython().system('pip install sklearn')
clear_output()


# In[ ]:





# In[44]:


import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from transformers import EvalPrediction
import torch
import tensorflow as tf
import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset
import datasets


# In[ ]:





# In[1]:


cols=[ 'admiration',
       'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
       'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
       'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
       'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
       'remorse', 'sadness', 'surprise', 'neutral']


# In[26]:


colss=["text",'label',"id"]


# In[ ]:


# df = pd.read_csv('/kaggle/input/go-emotions/goemotions_1.csv')


# In[28]:


train_dataset=pd.read_csv('/kaggle/input/new-go-emotion/train.tsv', sep='\t',names=colss, header=None)


# In[29]:


test_dataset=pd.read_csv('/kaggle/input/new-go-emotion/test.tsv', sep='\t',names=colss, header=None)


# In[36]:


validation_dataset=pd.read_csv('/kaggle/input/validation-go-emotion/dev.tsv', sep='\t',names=colss, header=None)


# In[ ]:





# In[62]:


labels_train=train_dataset['label']
labels_train=labels_train.to_numpy()
lab_train_int=[]
for entry in labels_train:
    ent=entry.split(',')
    lab_train_int.append(np.array(ent,dtype=int))
# print(lab_train_int)
mlb = MultiLabelBinarizer()
new_labels_train=mlb.fit_transform(lab_train_int)


# In[63]:


labels_test=test_dataset['label']
labels_test=labels_test.to_numpy()
lab_test_int=[]
for entry in labels_test:
    ent=entry.split(',')
    lab_test_int.append(np.array(ent,dtype=int))
# print(lab_train_int)
mlb = MultiLabelBinarizer()
new_labels_test=mlb.fit_transform(lab_test_int)


# In[64]:


labels_valid=validation_dataset['label']
labels_valid=labels_valid.to_numpy()
lab_valid_int=[]
for entry in labels_valid:
    ent=entry.split(',')
    lab_valid_int.append(np.array(ent,dtype=int))
# print(lab_train_int)
mlb = MultiLabelBinarizer()
new_labels_valid=mlb.fit_transform(lab_valid_int)


# In[65]:


train_dataset.drop('label',axis=1,inplace=True)
test_dataset.drop('label',axis=1,inplace=True)
validation_dataset.drop('label',axis=1,inplace=True)


# In[83]:


labels_train_one_hot=pd.DataFrame(new_labels_train,columns=cols)
labels_test_one_hot=pd.DataFrame(new_labels_test,columns=cols)
labels_valid_one_hot=pd.DataFrame(new_labels_valid,columns=cols)


# In[81]:


# labels_train_one_hot.head(10)


# In[88]:


training_data_op=pd.concat([train_dataset,labels_train_one_hot],axis=1)
testing_data_op=pd.concat([test_dataset,labels_test_one_hot],axis=1)
validation_data_op=pd.concat([validation_dataset,labels_valid_one_hot],axis=1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[13]:


#  test_dataset


# In[ ]:


# df.head()


# In[ ]:


# train, test = train_test_split(df, train_size=0.8)


# In[ ]:


# train, validation=train_test_split(test, train_size=0.5)


# In[85]:


# train_dataset


# In[ ]:





# In[89]:


hg_dataset_train = Dataset(pa.Table.from_pandas(training_data_op))
hg_dataset_test = Dataset(pa.Table.from_pandas(testing_data_op))
hg_dataset_valid = Dataset(pa.Table.from_pandas(validation_data_op))


# In[ ]:





# In[ ]:





# In[90]:


dataset={}
dataset['train']=hg_dataset_train
dataset['test']=hg_dataset_test
dataset['validation']=hg_dataset_valid


# In[91]:


dataset = datasets.DatasetDict(dataset)


# In[92]:


dataset


# In[18]:


type(dataset)


# In[ ]:





# In[ ]:


### convert to Huggingface dataset


# In[93]:


labels = [label for label in dataset['train'].features.keys() if label not in ['text', 'id']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
labels#,id2label,label2id


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# dataset


# In[ ]:


# df.columns


# In[ ]:


# counter=0
# id2label={}
# label2id={}
# label=['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
# for lab in ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']:
#     id2label[str(counter)]=lab
#     label2id[lab]=counter
#     counter+=1


# In[ ]:





# In[ ]:


# data=df['text'].copy()
# labels=df[['admiration',
#        'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
#        'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
#        'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']].copy()


# In[ ]:


# data=data.to_numpy()
# labels=labels.to_numpy()


# In[ ]:


# data.shape,labels.shape


# In[ ]:


# x_train, x_test, y_train, y_test = train_test_split(data,labels, train_size=0.8, random_state=0)


# In[ ]:


# x_test, x_valid, y_test, y_valid = train_test_split(x_test,y_test, train_size=0.5, random_state=0)


# In[ ]:


# x_train.shape


# In[94]:


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# In[95]:


# from transformers import AutoTokenizer
# import numpy as np

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_data(examples):
  # take a batch of texts
  text = examples["text"]
  # encode them
  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
  # add labels
  labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
  # create numpy array of shape (batch_size, num_labels)
  labels_matrix = np.zeros((len(text), len(labels)))
  # fill numpy array
  for idx, label in enumerate(labels):
    labels_matrix[:, idx] = labels_batch[label]

  encoding["labels"] = labels_matrix.tolist()
  
  return encoding


# In[ ]:


# encoding_train = tokenizer(list(x_train), padding="max_length", truncation=True, max_length=128)
# encoding_test = tokenizer(list(x_test), padding="max_length", truncation=True, max_length=128)
# encoding_valid = tokenizer(list(x_valid), padding="max_length", truncation=True, max_length=128)


# In[96]:


encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)


# In[97]:


encoded_dataset


# In[98]:


example = encoded_dataset['train'][0]
print(example)


# In[99]:


tokenizer.decode(example['input_ids'])


# In[ ]:





# In[ ]:


# def fun(data,labels):
#     _data=[]
    
#     for index in range(len(data["input_ids"])):
#         new_data={}
#         new_data["input_ids"]=data["input_ids"][index]
#         new_data["token_type_ids"]=data["token_type_ids"][index]
#         new_data["attention_mask"]=data["attention_mask"][index]
#         new_data["labels"]=labels[index]
#         _data.append(new_data)
#     return _data    


# In[ ]:


# training_data=fun(encoding_train, y_train)
# testing_data=fun(encoding_test, y_test)
# validation_data=fun(encoding_valid, y_valid)


# In[ ]:


# training_data[0]


# In[ ]:


# tf.convert_to_tensor(training_data)#, dtype=tf.float32


# In[ ]:





# ### I need to put our data into the format that is finally accepted by the bert
# 

# In[ ]:


# training_data.set_format("torch")
# encoding_test.set_format("torch")
# encoding_valid.set_format("torch")


# In[100]:


encoded_dataset.set_format("torch")


# # Model Defining

# In[101]:


model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=28,
                                                           id2label=id2label,
                                                           label2id=label2id,
                                                           hidden_dropout_prob = 0.1
                                                          )


# In[102]:


batch_size = 16
metric_name = "f1"


# In[104]:


args = TrainingArguments(
    f"bert-finetuned-sem_eval-english",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=4,#4
    weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model=metric_name,
    #push_to_hub=True,
)


# In[105]:


def multi_label_metrics(predictions, labels, threshold=0.3):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average=None)
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    precision = precision_score(y_true=y_true, y_pred=y_pred, average=None)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average=None)
    roc_auc = roc_auc_score(y_true, y_pred, average = 'macro')
    accuracy = accuracy_score(y_true, y_pred)
    
#     f1=f1.tolist()
#     precision=precision.tolist()
#     recall=recall.tolist()
#     f1=0
#     precision=0
#     recall=0
    
    # return as dictionary
    metrics = {'f1_macro': f1_macro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy,
               'f1' : json.dumps(f1.tolist()),
               'Precision' : json.dumps(precision.tolist()),
               'Recall' : json.dumps(recall.tolist())
              }
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result


# In[ ]:





# In[106]:


#forward pass
outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0), labels=encoded_dataset['train'][0]['labels'].unsqueeze(0))
outputs


# In[107]:


trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# In[108]:


trainer.train()


# In[ ]:


# API Key
#1daa61ad9ecca990d13db89f54ae5da1d6d0e458


# In[ ]:


# trainer.evaluate()


# In[ ]:


# for i in range(20):
#     print("Iteration : ",i," ---------------------------------------------------------------------------------------------------------------------------")
#     trainer.train()
#     print(trainer.evaluate())


# In[ ]:




