import pandas as pd
import numpy as np
import torch
import time
import torch.nn as nn 
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, random_split
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from collections import Counter
'''
data = pd.read_csv('train.csv').drop(columns=['id'])
print('Data read.')

#Pre-processing
data = data[data['POI/street'] != '/']
data.reset_index(inplace=True)
data = data.drop(columns=['index'])

data = pd.concat([data,pd.DataFrame(data['POI/street'].str.split('/',1).to_list(),columns=['POI','street'])],axis=1).drop(columns=['POI/street'])


def make_token(row, with_special=True):
    return tokenizer.tokenize(row, add_special_tokens=with_special, padding='max_length',max_length=MAX_LEN)


data['tokens'] = data.apply(lambda x: make_token(x['raw_address']),axis=1)
data['POI tokens'] = data.apply(lambda x: make_token(x['POI'], with_special=False),axis=1)
data['street tokens'] = data.apply(lambda x: make_token(x['street'], with_special=False),axis=1)
print('Data tokenized.')


def make_label(row):
    y = ['None' if token == '[PAD]' else 'POI' if token in row['POI tokens'] else 'Street' if token in row['street tokens'] else 'O' for token in row['tokens']]
    return y


data['labels'] = data.apply(lambda x: make_label(x),axis=1)
data.to_pickle('new_train.pkl')
'''
data = pd.read_pickle('new_train.pkl')
data_cleaned = data.drop(columns=['raw_address','POI','street','POI tokens','street tokens'])

#convert to numeric labels
tags = ['None','O','B_POI','I_POI','B_Street','I_Street']
CLASS = len(tags)
idxtotags = {i: t for i,t in enumerate(tags)}
tagstoidx = {t: i for i,t in idxtotags.items()}


def proper_tag(labels):
    label = []
    Pcounter = 0
    Scounter = 0
    for item in labels:
        if item == 'POI':
            if Pcounter == 0:
                Pcounter +=1
                label.append(tagstoidx['B_'+item])
            else:
                label.append(tagstoidx['I_'+item])
        elif item == 'Street':
                if Scounter == 0:
                    Scounter +=1
                    label.append(tagstoidx['B_'+item])
                else:
                    label.append(tagstoidx['I_'+item])
        else:
            label.append(tagstoidx[item])
            Scounter = 0
            Pcounter = 0
    return label

#import tokenizer
MAX_LEN = 64
tokenizer = AutoTokenizer.from_pretrained("sarahlintang/IndoBERT")
print('Tokenizer loaded.')

data_cleaned['labels'] = data_cleaned.apply(lambda x: proper_tag(x['labels']), axis=1)

encoded = tokenizer(list(data['raw_address']), add_special_tokens=True, padding='max_length',max_length=MAX_LEN)

inputs = torch.tensor(encoded['input_ids'])
attn = torch.tensor(encoded['attention_mask'])
labels = torch.tensor(data_cleaned['labels'])


class aem(nn.Module):
    def __init__(self, no_class):
        super().__init__()
        self.bert = AutoModel.from_pretrained("sarahlintang/IndoBERT")
        self.net = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.BatchNorm1d(64),
            nn.ReLU(),  
            nn.Dropout(p=0.2),
            # nn.Linear(1024, 512),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),  
            # nn.Dropout(p=0.2),
            nn.Linear(512, no_class),
            nn.GELU(),
        )
        
    
    def forward(self, inputs, attn):
        hidden = self.bert(inputs, token_type_ids=None, attention_mask=attn, return_dict = True)
        output = self.net(hidden[0])

        return output


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)


#initialise model
model = aem(CLASS).cuda(2)
# model.apply(init_weights)
print('Model initialised.')

BATCH_SIZE = 32
DROPOUT = 0.1
EPOCHS = 10
LR = 3e-5

data_set = TensorDataset(inputs,attn,labels)
train_set, val_set = random_split(data_set, [int(0.8*len(data_set))+1, int(0.2*len(data_set))])
train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, sampler = RandomSampler(train_set))
val_loader = DataLoader(val_set, batch_size = BATCH_SIZE, sampler = RandomSampler(val_set))

optimizer = AdamW(model.parameters(), lr = LR, eps = 1e-8)
total_step = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps = total_step)

WEIGHTS = torch.Tensor([0, 0.2, 1, 1, 1, 1]).cuda(2)


def loss(predicted, target):
    predicted = torch.rot90(predicted, -1, (1,2))
    criterion = nn.CrossEntropyLoss(weight = WEIGHTS,ignore_index=0, reduction='mean')
    return criterion(predicted, target)


def acc(predicted, actual):
    predicted_labels = torch.argmax(predicted, dim=2)
    #how many entities predicted
    correct_entities = torch.where((predicted_labels == actual) & (actual != 0) & (actual !=1), 1, 0)
    sum_correct_entities = torch.count_nonzero(correct_entities)
    total = torch.bincount(actual.view(-1))[2:].sum()

    #correct sentences predicted
    correct_predictions = torch.where((predicted_labels == actual) & (actual != 0), 1, 0).sum(dim=1)
    sentence_len = torch.count_nonzero(actual, dim=1)
    correct_sentences = sum(sentence_len == correct_predictions)
    return float(sum_correct_entities/total) * 100, int(correct_sentences)


train_losses = []
val_losses = []
val_accuracies = []


for epoch in range(EPOCHS):
    print('Transit to training.')
    model.train()
    
    total_train_loss = 0

    for step, batch in enumerate(train_loader):
        
        batch_inputs, batch_attn, batch_labels = batch
        batch_inputs = batch_inputs.type(torch.LongTensor).cuda(2)
        batch_attn = batch_attn.type(torch.LongTensor).cuda(2)
        batch_labels = batch_labels.type(torch.LongTensor).cuda(2)
        
        model.zero_grad()

        batch_output = model(batch_inputs, batch_attn).cuda(2)
        batch_loss = loss(batch_output, batch_labels).cuda(2)
        
        batch_loss.backward()
        total_train_loss += batch_loss
        
        nn.utils.clip_grad_norm_(parameters = model.parameters(), max_norm = 1)
        optimizer.step()
        scheduler.step()
        
        entities_pred, batch_acc = acc(batch_output, batch_labels)
        print(f'Epoch: {epoch}, Batch: {step}/{len(train_loader)} Current loss: {batch_loss:.2f}')
        print('Perc correct entities: {0:.2f} \nNo correct sentences: {1}'.format(entities_pred, batch_acc))
        
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print("Average train loss: {}".format(avg_train_loss))

    print("Transit to validation")
    
    model.eval()
    
    total_val_loss = 0
    total_val_acc = 0
    
    for step, batch in enumerate(val_loader):
        batch_inputs, batch_attn, batch_labels = batch
        batch_inputs = batch_inputs.type(torch.LongTensor).cuda(2)
        batch_attn = batch_attn.type(torch.LongTensor).cuda(2)
        batch_labels = batch_labels.type(torch.LongTensor).cuda(2)
        
        model.zero_grad()
        
        with torch.no_grad():
            val_output = model(batch_inputs, batch_attn)
            val_loss = loss(val_output, batch_labels).cuda(2)

            total_val_loss += val_loss

            val_acc, _ = acc(val_output, batch_labels)
            total_val_acc += val_acc
    
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print("Average validation loss: {}".format(avg_val_loss))

    avg_val_acc = total_val_acc / len(train_loader)
    val_accuracies.append(avg_val_loss)
    print("Average validation accuracy: {}".format(avg_val_acc))

torch.save(model.state_dict(), 'train_model')

plt.plot([*range(4)], train_losses)
plt.plot([*range(4)], val_losses)
plt.xlabel('Epochs')
labels = ['Train_loss','Val_loss']
plt.legend(labels)
plt.savefig('performance.png')