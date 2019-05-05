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


data_BBdiff = (ImageList.from_folder(path)
                .split_by_folder(train = "train_BBdiff", valid = "validation_BBdiff")
                .label_from_func(get_labels)
                .databunch().normalize(imagenet_stats)
            )


# In[10]:


data_BBdiff.show_batch(3, figsize=(9,6))


# In[11]:


learn_BBdiff = cnn_learner(data_BBdiff, models.resnet50, metrics = [r2_score, root_mean_squared_error])
learn_BBdiff.load('BBdiff-stage1');


# In[12]:


last_layers_BBdiff = list(children(learn_BBdiff.model))[-1][-1]
learn_BBdiff.model[-1] = learn_BBdiff.model[-1][:-1]
last_layers_BBdiff


# In[13]:


data_MA20050diff = (ImageList.from_folder(path)
                .split_by_folder(train = "train_MA20050diff", valid = "validation_MA20050diff")
                .label_from_func(get_labels)
                .databunch().normalize(imagenet_stats)
            )


# In[14]:


data_MA20050diff.show_batch(3, figsize=(9,6))


# In[15]:


learn_MA20050diff = cnn_learner(data_MA20050diff, models.resnet50, metrics = [r2_score, root_mean_squared_error])
learn_MA20050diff.load('MA20050diff-stage1');


# In[16]:


last_layers_MA20050diff = list(children(learn_MA20050diff.model))[-1][-1]
learn_MA20050diff.model[-1] = learn_MA20050diff.model[-1][:-1]
last_layers_MA20050diff


# In[17]:


data_MACD = (ImageList.from_folder(path)
                .split_by_folder(train = "train_MACD", valid = "validation_MACD")
                .label_from_func(get_labels)
                .databunch().normalize(imagenet_stats)
            )


# In[18]:


data_MACD.show_batch(3, figsize=(9,6))


# In[19]:


learn_MACD = cnn_learner(data_MACD, models.resnet50, metrics = [r2_score, root_mean_squared_error])
learn_MACD.load('MACD-stage1');


# In[20]:


last_layers_MACD = list(children(learn_MACD.model))[-1][-1]
learn_MACD.model[-1] = learn_MACD.model[-1][:-1]
last_layers_MACD


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


train_ds = ConcatDataset(data_percentage.train_ds.x, data_BBdiff.train_ds.x,
                         data_MA20050diff.train_ds.x, data_BBdiff.train_ds.x,
                         data_percentage.train_ds.y)
valid_ds = ConcatDataset(data_percentage.valid_ds.x, data_BBdiff.valid_ds.x,
                         data_MA20050diff.valid_ds.x, data_BBdiff.valid_ds.x,
                         data_percentage.valid_ds.y)


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


model = ConcatModel(learn_percentage.model, learn_BBdiff.model, learn_MA20050diff.model, learn_MACD.model, 
                    last_layers_percentage, last_layers_BBdiff, last_layers_MA20050diff, last_layers_MACD)


# In[30]:


layer_groups = [nn.Sequential(*flatten_model(learn_percentage.layer_groups[0])),
                nn.Sequential(*flatten_model(learn_percentage.layer_groups[1])),
                nn.Sequential(*flatten_model(learn_percentage.layer_groups[2])),
                nn.Sequential(*flatten_model(learn_BBdiff.layer_groups[0])),
                nn.Sequential(*flatten_model(learn_BBdiff.layer_groups[1])),
                nn.Sequential(*flatten_model(learn_BBdiff.layer_groups[2])),
                nn.Sequential(*flatten_model(learn_MA20050diff.layer_groups[0])),
                nn.Sequential(*flatten_model(learn_MA20050diff.layer_groups[1])),
                nn.Sequential(*flatten_model(learn_MA20050diff.layer_groups[2])),
                nn.Sequential(*flatten_model(learn_MACD.layer_groups[0])),
                nn.Sequential(*flatten_model(learn_MACD.layer_groups[1])),
                nn.Sequential(*flatten_model(learn_MACD.layer_groups[2]))] 


# In[31]:


learn = Learner(data, model, metrics= [r2_score, root_mean_squared_error], layer_groups=layer_groups)


# In[41]:


learn.fit_one_cycle(9)


# In[42]:


learn.save('percentage+BBdiff+MA20050diff+MACD-stage1')


# In[32]:


learn.load('percentage+BBdiff+MA20050diff+MACD-stage1');


# In[50]:


learn.fit_one_cycle(3, 0.001)


# In[51]:


learn.save('percentage+BBdiff+MA20050diff+MACD-stage2')


# In[52]:


learn.fit_one_cycle(3, 0.0003)


# In[53]:


learn.save('percentage+BBdiff+MA20050diff+MACD-stage3')


# In[33]:


learn.load('percentage+BBdiff+MA20050diff+MACD-stage3');


# In[33]:


learn.fit_one_cycle(6, 0.0003)


# In[35]:


learn.save('percentage+BBdiff+MA20050diff+MACD-stage4')


# In[34]:


learn.load('percentage+BBdiff+MA20050diff+MACD-stage4');


# https://docs.fast.ai/data_block.html#Add-a-test-set

# In[50]:


data_percentage_test = (ImageList.from_folder(path)
                .split_by_folder(train = "train_percentage", valid = "test_percentage")
                .label_from_func(get_labels)
                .databunch().normalize(imagenet_stats)
            )

data_BBdiff_test= (ImageList.from_folder(path)
                .split_by_folder(train = "train_BBdiff", valid = "test_BBdiff")
                .label_from_func(get_labels)
                .databunch().normalize(imagenet_stats)
            )

data_MA20050diff_test = (ImageList.from_folder(path)
                .split_by_folder(train = "train_MA20050diff", valid = "test_MA20050diff")
                .label_from_func(get_labels)
                .databunch().normalize(imagenet_stats)
            )

data_MACD_test = (ImageList.from_folder(path)
                .split_by_folder(train = "train_MACD", valid = "test_MACD")
                .label_from_func(get_labels)
                .databunch().normalize(imagenet_stats)
            )

train_ds = ConcatDataset(data_percentage_test.train_ds.x, data_BBdiff_test.train_ds.x,
                         data_MA20050diff_test.train_ds.x, data_MACD_test.train_ds.x,
                         data_percentage_test.train_ds.y)
test_ds = ConcatDataset(data_percentage_test.valid_ds.x, data_BBdiff_test.valid_ds.x,
                        data_MA20050diff_test.valid_ds.x, data_MACD_test.valid_ds.x,
                        data_percentage_test.valid_ds.y)

bs = 64
train_dl = DataLoader(train_ds, bs)
test_dl = DataLoader(test_ds, bs)
data_test = DataBunch(train_dl, test_dl)


# In[51]:


data_test.test_dl


# In[52]:


pred_metrics = learn.validate(data_test.test_dl)


# In[53]:


pred_metrics

