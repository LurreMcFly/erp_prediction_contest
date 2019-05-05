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


data_CLOSE = (ImageList.from_folder(path)
                .split_by_folder(train = "train_CLOSE", valid = "validation_CLOSE")
                .label_from_func(get_labels)
                .databunch().normalize(imagenet_stats)
            )


# In[6]:


data_CLOSE.show_batch(3, figsize=(9,6))


# In[7]:


learn_CLOSE = cnn_learner(data_CLOSE, models.resnet50, metrics = [r2_score, root_mean_squared_error])
learn_CLOSE.load('CLOSE-stage1');


# In[8]:


last_layers_CLOSE = list(children(learn_CLOSE.model))[-1][-1]
learn_CLOSE.model[-1] = learn_CLOSE.model[-1][:-1]
last_layers_CLOSE


# In[9]:


data_percentage = (ImageList.from_folder(path)
                .split_by_folder(train = "train_percentage", valid = "validation_percentage")
                .label_from_func(get_labels)
                .databunch().normalize(imagenet_stats)
            )


# In[10]:


data_percentage.show_batch(3, figsize=(9,6))


# In[11]:


learn_percentage = cnn_learner(data_percentage, models.resnet50, metrics = [r2_score, root_mean_squared_error])
learn_percentage.load('percentage-stage1');


# In[12]:


last_layers_percentage = list(children(learn_percentage.model))[-1][-1]
learn_percentage.model[-1] = learn_percentage.model[-1][:-1]
last_layers_percentage


# In[13]:


data_RSI = (ImageList.from_folder(path)
                .split_by_folder(train = "train_RSI", valid = "validation_RSI")
                .label_from_func(get_labels)
                .databunch().normalize(imagenet_stats)
            )


# In[14]:


data_RSI.show_batch(3, figsize=(9,6))


# In[15]:


learn_RSI = cnn_learner(data_RSI, models.resnet50, metrics = [r2_score, root_mean_squared_error])
learn_RSI.load('RSI-stage1');


# In[16]:


last_layers_RSI = list(children(learn_RSI.model))[-1][-1]
learn_RSI.model[-1] = learn_RSI.model[-1][:-1]
last_layers_RSI


# In[17]:


data_BBdiff = (ImageList.from_folder(path)
                .split_by_folder(train = "train_BBdiff", valid = "validation_BBdiff")
                .label_from_func(get_labels)
                .databunch().normalize(imagenet_stats)
            )


# In[18]:


data_BBdiff.show_batch(3, figsize=(9,6))


# In[19]:


learn_BBdiff = cnn_learner(data_BBdiff, models.resnet50, metrics = [r2_score, root_mean_squared_error])
learn_BBdiff.load('BBdiff-stage1');


# In[20]:


last_layers_BBdiff = list(children(learn_BBdiff.model))[-1][-1]
learn_BBdiff.model[-1] = learn_BBdiff.model[-1][:-1]
last_layers_BBdiff


# In[21]:


class ConcatDataset(Dataset):
    def __init__(self, x1, x2, x3, x4, y): 
        self.x1,self.x2,self.x3,self.x4,self.y = x1,x2,x3,x4,y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): 
        imgTensor1 = self.x1[i].data
        imgTensor2 = self.x2[i].data
        imgTensor3 = self.x3[i].data
        imgTensor4 = self.x4[i].data
        y = torch.from_numpy(self.y[i].data)
        
        return (imgTensor1, imgTensor2, imgTensor3, imgTensor4), y


# In[22]:


train_ds = ConcatDataset(data_CLOSE.train_ds.x, data_percentage.train_ds.x,
                         data_RSI.train_ds.x, data_BBdiff.train_ds.x,
                         data_CLOSE.train_ds.y)
valid_ds = ConcatDataset(data_CLOSE.valid_ds.x, data_percentage.valid_ds.x,
                         data_RSI.valid_ds.x, data_BBdiff.valid_ds.x,
                         data_CLOSE.valid_ds.y)


# In[23]:


bs = 64
train_dl = DataLoader(train_ds, bs)
valid_dl = DataLoader(valid_ds, bs)
data = DataBunch(train_dl, valid_dl)


# In[24]:


data.train_ds.x1[0]


# In[25]:


data.train_ds.x2[0]


# In[26]:


data.train_ds.x3[0]


# In[27]:


data.train_ds.x4[0]


# In[28]:


class ConcatModel(nn.Module):
    def __init__(self, cnn_1, cnn_2, cnn_3, cnn_4, last_layers_1, last_layers_2, last_layers_3, last_layers_4):
        super().__init__()
        self.cnn_1 = cnn_1
        self.cnn_2 = cnn_2
        self.cnn_3 = cnn_3
        self.cnn_4 = cnn_4
               
        self.last_layers = nn.Linear(4*512, 1, True)
        self.last_layers.weight[0].data[0:512] = last_layers_1.weight[0].data
        self.last_layers.weight[0].data[512:2*512] = last_layers_2.weight[0].data
        self.last_layers.weight[0].data[2*512:3*512] = last_layers_3.weight[0].data
        self.last_layers.weight[0].data[3*512:4*512] = last_layers_4.weight[0].data
        
    def forward(self, x1, x2, x3, x4):
        x1 = self.cnn_1(x1)
        x2 = self.cnn_2(x2)
        x3 = self.cnn_3(x3)
        x4 = self.cnn_4(x4)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.last_layers(x)
        
        return x


# In[29]:


model = ConcatModel(learn_CLOSE.model, learn_percentage.model, learn_RSI.model, learn_BBdiff.model, 
                    last_layers_CLOSE, last_layers_percentage, last_layers_RSI, last_layers_BBdiff)


# In[30]:


layer_groups = [nn.Sequential(*flatten_model(learn_CLOSE.layer_groups[0])),
                nn.Sequential(*flatten_model(learn_CLOSE.layer_groups[1])),
                nn.Sequential(*flatten_model(learn_CLOSE.layer_groups[2])),
                nn.Sequential(*flatten_model(learn_percentage.layer_groups[0])),
                nn.Sequential(*flatten_model(learn_percentage.layer_groups[1])),
                nn.Sequential(*flatten_model(learn_percentage.layer_groups[2])),
                nn.Sequential(*flatten_model(learn_RSI.layer_groups[0])),
                nn.Sequential(*flatten_model(learn_RSI.layer_groups[1])),
                nn.Sequential(*flatten_model(learn_RSI.layer_groups[2])),
                nn.Sequential(*flatten_model(learn_BBdiff.layer_groups[0])),
                nn.Sequential(*flatten_model(learn_BBdiff.layer_groups[1])),
                nn.Sequential(*flatten_model(learn_BBdiff.layer_groups[2]))] 


# In[31]:


learn = Learner(data, model, metrics= [r2_score, root_mean_squared_error], layer_groups=layer_groups)


# In[32]:


learn.fit_one_cycle(9)


# In[33]:


learn.save('CLOSE+percentage+RSI+BBdiff-stage1')


# In[34]:


learn.fit_one_cycle(3)


# In[35]:


learn.save('CLOSE+percentage+RSI+BBdiff-stage2')


# In[37]:


learn.lr_find()
learn.recorder.plot()


# In[38]:


learn.fit_one_cycle(3)


# In[39]:


learn.lr_find()
learn.recorder.plot()


# In[40]:


learn.save('CLOSE+percentage+RSI+BBdiff-stage3')


# In[41]:


learn.fit_one_cycle(6, max_lr = slice(3e-6,3e-4))


# In[42]:


learn.save('CLOSE+percentage+RSI+BBdiff-stage4')


# In[32]:


learn.load('CLOSE+percentage+RSI+BBdiff-stage4');


# https://docs.fast.ai/data_block.html#Add-a-test-set

# In[47]:


data_CLOSE_test = (ImageList.from_folder(path)
                .split_by_folder(train = "train_CLOSE", valid = "test_CLOSE")
                .label_from_func(get_labels)
                .databunch().normalize(imagenet_stats)
            )

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

data_BBdiff_test = (ImageList.from_folder(path)
                .split_by_folder(train = "train_BBdiff", valid = "test_BBdiff")
                .label_from_func(get_labels)
                .databunch().normalize(imagenet_stats)
            )

train_ds = ConcatDataset(data_CLOSE_test.train_ds.x, data_percentage_test.train_ds.x,
                         data_RSI_test.train_ds.x, data_BBdiff_test.train_ds.x,
                         data_CLOSE_test.train_ds.y)
test_ds = ConcatDataset(data_CLOSE_test.valid_ds.x, data_percentage_test.valid_ds.x,
                        data_RSI_test.valid_ds.x, data_BBdiff_test.valid_ds.x,
                        data_CLOSE_test.valid_ds.y)

bs = 64
train_dl = DataLoader(train_ds, bs)
test_dl = DataLoader(test_ds, bs)
data_test = DataBunch(train_dl, test_dl)


# In[48]:


data_test.test_dl


# In[49]:


pred_metrics = learn.validate(data_test.test_dl)


# In[50]:


pred_metrics

