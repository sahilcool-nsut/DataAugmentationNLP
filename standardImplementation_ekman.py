#!/usr/bin/env python
# coding: utf-8

# In[4]:


from IPython.display import clear_output


# In[5]:


from IPython.display import clear_output 
get_ipython().system('python3 --version')


# In[6]:


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


# In[7]:


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


# In[8]:


cols=[ 'admiration',
       'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
       'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
       'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
       'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
       'remorse', 'sadness', 'surprise', 'neutral']


# In[9]:


cols_drop=[ 'id','admiration',
       'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
       'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
       'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
       'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
       'remorse', 'sadness', 'surprise']


# In[10]:


colss=["text",'label',"id"]


# In[11]:


train_dataset=pd.read_csv('/kaggle/input/new-go-emotion/train.tsv', sep='\t',names=colss, header=None)


# In[12]:


test_dataset=pd.read_csv('/kaggle/input/new-go-emotion/test.tsv', sep='\t',names=colss, header=None)


# In[13]:


validation_dataset=pd.read_csv('/kaggle/input/validation-go-emotion/dev.tsv', sep='\t',names=colss, header=None)


# In[14]:


labels_train=train_dataset['label']
labels_train=labels_train.to_numpy()
lab_train_int=[]
for entry in labels_train:
    ent=entry.split(',')
    lab_train_int.append(np.array(ent,dtype=int))
# print(lab_train_int)
mlb = MultiLabelBinarizer()
new_labels_train=mlb.fit_transform(lab_train_int)


# In[15]:


labels_test=test_dataset['label']
labels_test=labels_test.to_numpy()
lab_test_int=[]
for entry in labels_test:
    ent=entry.split(',')
    lab_test_int.append(np.array(ent,dtype=int))
# print(lab_train_int)
mlb = MultiLabelBinarizer()
new_labels_test=mlb.fit_transform(lab_test_int)


# In[16]:


labels_valid=validation_dataset['label']
labels_valid=labels_valid.to_numpy()
lab_valid_int=[]
for entry in labels_valid:
    ent=entry.split(',')
    lab_valid_int.append(np.array(ent,dtype=int))
# print(lab_train_int)
mlb = MultiLabelBinarizer()
new_labels_valid=mlb.fit_transform(lab_valid_int)


# In[17]:


train_dataset.drop('label',axis=1,inplace=True)
test_dataset.drop('label',axis=1,inplace=True)
validation_dataset.drop('label',axis=1,inplace=True)


# In[18]:


labels_train_one_hot=pd.DataFrame(new_labels_train,columns=cols)
labels_test_one_hot=pd.DataFrame(new_labels_test,columns=cols)
labels_valid_one_hot=pd.DataFrame(new_labels_valid,columns=cols)


# In[19]:


training_data_op=pd.concat([train_dataset,labels_train_one_hot],axis=1)
testing_data_op=pd.concat([test_dataset,labels_test_one_hot],axis=1)
validation_data_op=pd.concat([validation_dataset,labels_valid_one_hot],axis=1)


# In[ ]:





# In[20]:


training_data_op['anger_'] = training_data_op.anger | training_data_op.annoyance | training_data_op.disapproval
training_data_op['disgust_'] = training_data_op.disgust
training_data_op['fear_'] = training_data_op.fear | training_data_op.nervousness
training_data_op['joy_'] = training_data_op.joy | training_data_op.amusement | training_data_op.admiration | training_data_op.approval | training_data_op.caring | training_data_op.desire | training_data_op.excitement | training_data_op.gratitude | training_data_op.love | training_data_op.optimism | training_data_op.pride | training_data_op.relief
training_data_op['sadness_'] = training_data_op.sadness | training_data_op.disappointment | training_data_op.embarrassment | training_data_op.grief | training_data_op.remorse
training_data_op['surprise_'] = training_data_op.surprise | training_data_op.confusion | training_data_op.curiosity | training_data_op.realization
training_data_op.drop(cols_drop,axis=1,inplace=True)


# In[21]:


# training_data_op


# In[22]:


testing_data_op['anger_'] = testing_data_op.anger | testing_data_op.annoyance | testing_data_op.disapproval
testing_data_op['disgust_'] = testing_data_op.disgust
testing_data_op['fear_'] = testing_data_op.fear | testing_data_op.nervousness
testing_data_op['joy_'] = testing_data_op.joy | testing_data_op.amusement | testing_data_op.admiration | testing_data_op.approval | testing_data_op.caring | testing_data_op.desire | testing_data_op.excitement | testing_data_op.gratitude | testing_data_op.love | testing_data_op.optimism | testing_data_op.pride | testing_data_op.relief
testing_data_op['sadness_'] = testing_data_op.sadness | testing_data_op.disappointment | testing_data_op.embarrassment | testing_data_op.grief | testing_data_op.remorse
testing_data_op['surprise_'] = testing_data_op.surprise | testing_data_op.confusion | testing_data_op.curiosity | testing_data_op.realization
testing_data_op.drop(cols_drop,axis=1,inplace=True)


# In[23]:


validation_data_op['anger_'] = validation_data_op.anger | validation_data_op.annoyance | validation_data_op.disapproval
validation_data_op['disgust_'] = validation_data_op.disgust
validation_data_op['fear_'] = validation_data_op.fear | validation_data_op.nervousness
validation_data_op['joy_'] = validation_data_op.joy | validation_data_op.amusement | validation_data_op.admiration | validation_data_op.approval | validation_data_op.caring | validation_data_op.desire | validation_data_op.excitement | validation_data_op.gratitude | validation_data_op.love | validation_data_op.optimism | validation_data_op.pride | validation_data_op.relief
validation_data_op['sadness_'] = validation_data_op.sadness | validation_data_op.disappointment | validation_data_op.embarrassment | validation_data_op.grief | validation_data_op.remorse
validation_data_op['surprise_'] = validation_data_op.surprise | validation_data_op.confusion | validation_data_op.curiosity | validation_data_op.realization
validation_data_op.drop(cols_drop,axis=1,inplace=True)


# In[24]:


hg_dataset_train = Dataset(pa.Table.from_pandas(training_data_op))
hg_dataset_test = Dataset(pa.Table.from_pandas(testing_data_op))
hg_dataset_valid = Dataset(pa.Table.from_pandas(validation_data_op))


# In[25]:


dataset={}
dataset['train']=hg_dataset_train
dataset['test']=hg_dataset_test
dataset['validation']=hg_dataset_valid


# In[26]:


dataset = datasets.DatasetDict(dataset)


# In[27]:


dataset


# In[28]:


type(dataset)


# In[29]:


labels = [label for label in dataset['train'].features.keys() if label not in ['text', 'id']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
labels#,id2label,label2id


# In[30]:


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# In[31]:


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


# In[32]:


encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)


# In[33]:


encoded_dataset


# In[34]:


example = encoded_dataset['train'][0]
print(example)


# In[35]:


tokenizer.decode(example['input_ids'])


# In[36]:


encoded_dataset.set_format("torch")


# In[37]:


model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=7,
                                                           id2label=id2label,
                                                           label2id=label2id,
                                                           hidden_dropout_prob = 0.1
                                                          )


# In[38]:


batch_size = 16
metric_name = "f1"


# In[39]:


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


# In[40]:


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


# In[41]:


#forward pass
outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0), labels=encoded_dataset['train'][0]['labels'].unsqueeze(0))
outputs


# In[42]:


trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# In[43]:


trainer.train()


# In[44]:


#Api key
#1daa61ad9ecca990d13db89f54ae5da1d6d0e458


# In[45]:


trainer.evaluate()


# In[ ]:




