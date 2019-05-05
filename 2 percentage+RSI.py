#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate


# In[3]:


path = Path('images'); path


# In[4]:


def get_labels(file_path): return float(str(file_path).split(",")[1].split(".p")[0])


# In[5]:


data_percentage = (ImageList.from_folder(path)
                .split_by_folder(train = "train_percentage", valid = "validation_percentage")
                .label_from_func(get_labels)
                .databunch().normalize(imagenet_stats)
            )


# In[6]:


data_percentage.show_batch(3, figsize=(9,6))


# In[7]:


learn_percentage = cnn_learner(data_percentage, models.resnet50, metrics = [r2_score, root_mean_squared_error])
learn_percentage.load('percentage-stage1');


# In[8]:


last_layers_percentage = list(children(learn_percentage.model))[-1][-1]
learn_percentage.model[-1] = learn_percentage.model[-1][:-1]
last_layers_percentage


# In[9]:


data_RSI = (ImageList.from_folder(path)
            .split_by_folder(train = "train_RSI", valid = "validation_RSI")
            .label_from_func(get_labels)
            .databunch().normalize(imagenet_stats)
        )


# In[10]:


data_RSI.show_batch(3, figsize=(9,6))


# In[11]:


learn_RSI = cnn_learner(data_RSI, models.resnet50, metrics = [r2_score, root_mean_squared_error])
learn_RSI.load('RSI-stage1');


# In[12]:


last_layers_RSI = list(children(learn_RSI.model))[-1][-1]
learn_RSI.model[-1] = learn_RSI.model[-1][:-1]


# In[13]:


class ConcatDataset(Dataset):
    def __init__(self, x1, x2, y): self.x1,self.x2,self.y = x1,x2,y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): 
        imgTensor1 = self.x1[i].data#.contiguous()
        imgTensor2 = self.x2[i].data#.contiguous()
        y = torch.from_numpy(self.y[i].data)
        #y =  torch.from_numpy(self.y[i].data).contiguous()
        
        return (imgTensor1, imgTensor2), y


# In[14]:


train_ds = ConcatDataset(data_percentage.train_ds.x, data_RSI.train_ds.x, data_percentage.train_ds.y)
valid_ds = ConcatDataset(data_percentage.valid_ds.x, data_RSI.valid_ds.x, data_percentage.valid_ds.y)


# In[15]:


bs = 64
train_dl = DataLoader(train_ds, bs)
valid_dl = DataLoader(valid_ds, bs)
data = DataBunch(train_dl, valid_dl)


# In[16]:


data.train_ds.x1[0]


# In[17]:


data.train_ds.x2[0]


# In[18]:


class ConcatModel(nn.Module):
    def __init__(self, cnn_1, cnn_2, last_layers_1, last_layers_2):
        super().__init__()
        self.cnn_1 = cnn_1
        self.cnn_2 = cnn_2
        
        self.last_layers = nn.Linear(2*512, 1, True)
        self.last_layers.weight[0].data[0:512] = last_layers_1.weight[0].data
        self.last_layers.weight[0].data[512:2*512] = last_layers_2.weight[0].data
        
        
    def forward(self, x1, x2):
        x1 = self.cnn_1(x1)
        x2 = self.cnn_2(x2)
        
        x = torch.cat((x1, x2), dim=1)
        x = self.last_layers(x)
        
        return x


# In[19]:


model = ConcatModel(learn_percentage.model, learn_RSI.model, last_layers_percentage, last_layers_RSI)


# In[20]:


layer_groups = [nn.Sequential(*flatten_model(learn_percentage.layer_groups[0])),
                nn.Sequential(*flatten_model(learn_percentage.layer_groups[1])),
                nn.Sequential(*flatten_model(learn_percentage.layer_groups[2])),
                nn.Sequential(*flatten_model(learn_RSI.layer_groups[0])),
                nn.Sequential(*flatten_model(learn_RSI.layer_groups[1])),
                nn.Sequential(*flatten_model(learn_RSI.layer_groups[2]))] 


# In[21]:


learn = Learner(data, model, metrics= [r2_score, root_mean_squared_error], layer_groups=layer_groups)


# In[22]:


learn.fit_one_cycle(9)


# In[23]:


learn.save('percentage+RSI-stage1')


# In[22]:


learn.load('percentage+RSI-stage1');


# In[23]:


learn.lr_find()
learn.recorder.plot()


# In[24]:


learn.fit_one_cycle(9, max_lr = slice(3e-6,3e-5))


# In[25]:


learn.save('percentage+RSI-stage2')


# In[23]:


learn.load('percentage+RSI-stage2');


# https://docs.fast.ai/data_block.html#Add-a-test-set

# In[28]:


data_percentage_test = (ImageList.from_folder(path)
                .split_by_folder(train = "train_percentage", valid = "test_percentage")
                .label_from_func(get_labels)
                .databunch().normalize(imagenet_stats)
            )

data_RSI_test = (ImageList.from_folder(path)
                .split_by_folder(train = "train_RSI", valid = "test_RSI")
                .label_from_func(get_labels)
                .databunch().normalize(imagenet_stats)
            )

train_ds = ConcatDataset(data_percentage_test.train_ds.x, data_RSI_test.train_ds.x, data_percentage_test.train_ds.y)
test_ds = ConcatDataset(data_percentage_test.valid_ds.x, data_RSI_test.valid_ds.x, data_percentage_test.valid_ds.y)

bs = 64
train_dl = DataLoader(train_ds, bs)
test_dl = DataLoader(test_ds, bs)
data_test = DataBunch(train_dl, test_dl)


# In[29]:


data_test.test_dl


# In[30]:


pred_metrics = learn.validate(data_test.test_dl)


# In[31]:


pred_metrics

