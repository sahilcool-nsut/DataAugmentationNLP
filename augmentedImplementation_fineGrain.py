#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.display import clear_output 


# In[3]:


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
clear_output()


# In[81]:


import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from transformers import EvalPrediction
import torch
import tensorflow as tf
import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset
import datasets


# In[10]:


cols=[ 'admiration',
       'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
       'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
       'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
       'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
       'remorse', 'sadness', 'surprise', 'neutral']


# In[11]:


colss=["text",'label',"id"]


# In[12]:


# df = pd.read_csv('/kaggle/input/augmented-data-original/augmented_data (2).csv')


# In[25]:


training_dataset = pd.read_csv('/kaggle/input/augmented-data-original/augmented_data (2).csv')


# In[38]:


test_dataset=pd.read_csv('/kaggle/input/new-go-emotion/test.tsv', sep='\t',names=colss, header=None)


# In[39]:


labels_test=test_dataset['label']
labels_test=labels_test.to_numpy()
lab_test_int=[]
for entry in labels_test:
    ent=entry.split(',')
    lab_test_int.append(np.array(ent,dtype=int))
# print(lab_train_int)
mlb = MultiLabelBinarizer()
new_labels_test=mlb.fit_transform(lab_test_int)


# In[40]:


test_dataset.drop('label',axis=1,inplace=True)
labels_test_one_hot=pd.DataFrame(new_labels_test,columns=cols)
testing_data_op=pd.concat([test_dataset,labels_test_one_hot],axis=1)


# In[31]:


training_dataset.drop('Unnamed: 0',axis=1,inplace=True)


# In[50]:


training_dataset.rename(columns = {'Text_Data':'text'}, inplace = True)


# In[51]:


training_dataset


# In[41]:


testing_data_op.drop('id',axis=1,inplace=True)


# In[52]:


testing_data_op


# In[ ]:





# In[7]:


# df.drop('Unnamed: 0',inplace=True,axis=1)


# In[9]:


# df.head()


# In[9]:


# train, test = train_test_split(df, train_size=0.8)


# In[53]:


hg_dataset_train = Dataset(pa.Table.from_pandas(training_dataset))
hg_dataset_test = Dataset(pa.Table.from_pandas(testing_data_op))
# hg_dataset_valid = Dataset(pa.Table.from_pandas(validation))


# In[54]:


dataset={}
dataset['train']=hg_dataset_train
dataset['test']=hg_dataset_test
# dataset['validation']=hg_dataset_valid


# In[55]:


dataset = datasets.DatasetDict(dataset)


# In[56]:


dataset


# In[ ]:





# In[57]:


type(dataset)


# In[63]:


labels = [label for label in dataset['train'].features.keys() if label not in ['text']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
labels#,id2label,label2id


# In[64]:


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# In[65]:


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


# In[66]:


encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)


# In[67]:


encoded_dataset


# In[70]:


example = encoded_dataset['train'][0]
print(example)


# In[71]:


tokenizer.decode(example['input_ids'])


# In[72]:


encoded_dataset.set_format("torch")


# In[73]:


model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=28,
                                                           id2label=id2label,
                                                           label2id=label2id)


# In[74]:


batch_size = 16
metric_name = "f1"


# In[75]:


args = TrainingArguments(
    f"bert-finetuned-sem_eval-english",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=4,
    weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model=metric_name,
    #push_to_hub=True,
)


# In[77]:


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


# In[78]:


#forward pass
outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0), labels=encoded_dataset['train'][0]['labels'].unsqueeze(0))
outputs


# In[79]:


trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# In[80]:


trainer.train()


# In[30]:


#Api key
#1daa61ad9ecca990d13db89f54ae5da1d6d0e458


# In[32]:


trainer.evaluate()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




