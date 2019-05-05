#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[3]:


bs = 64


# In[4]:


path = Path('images'); path


# In[5]:


np.random.seed(2)
def get_labels(file_path): return float(str(file_path).split(",")[1].split(".p")[0])


# In[6]:


data = (ImageList.from_folder(path)
                .split_by_folder(train = "train_percentage", valid = "validation_percentage",)
                .label_from_func(get_labels)
                .add_test_folder(test_folder = "test_percentage")
                .databunch().normalize(imagenet_stats)
            )


# In[7]:


data.show_batch(3, figsize=(9,6))


# In[8]:


learn = cnn_learner(data, models.resnet50, metrics = [r2_score, root_mean_squared_error])


# In[9]:


learn.unfreeze()


# In[10]:


learn.fit_one_cycle(9)


# In[11]:


learn.save('percentage-stage1')


# In[10]:


learn.load('percentage-stage1')


# In[15]:


data_test = (ImageList.from_folder(path)
                .split_by_folder(train = "train_percentage", valid = "test_percentage",)
                .label_from_func(get_labels)
                .databunch().normalize(imagenet_stats)
            )


# In[16]:


pred_metrics = learn.validate(data_test.test_dl)


# In[17]:


pred_metrics

