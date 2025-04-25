#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['GEMINI_API_KEY'] = 'AIzaSyB0NXK3cDwhotSiEFKbChqKAlkshdyc5x4'


# In[2]:


import google.generativeai as genai
genai.configure(api_key=os.environ['GEMINI_API_KEY'])


# In[3]:


#model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')


# In[4]:


response = model.generate_content("What would be the future scope of generative AI in 2030?")


# In[5]:


import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown

def to_markdown(text):
    text= text.replace('.',' *')
    return Markdown(textwrap.indent(text,'> ',predicate=lambda _:True))


# In[6]:


for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)


# In[7]:


#model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")


# In[8]:


get_ipython().run_cell_magic('time', '', 'response = model.generate_content("create a table for summarize to a kid about datascince vs machine learning vs nlp vs deep learning vs neural network vs generative AI vs agentic ai vs mcp vs agi and save this generated file to my computer as a pdf file")\n')


# In[9]:


to_markdown(response.text)


# In[10]:


for chunk in response:
    print(chunk.text)
    print("_"*5)


# In[11]:


import PIL.Image
img=PIL.Image.open(r'/Users/aviswe/Desktop/830/Google Gemini.jpeg')



# In[12]:


model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")


# In[13]:


img


# In[15]:


response = model.generate_content(img)
to_markdown(response.text)


# In[ ]:




