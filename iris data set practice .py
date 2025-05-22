#!/usr/bin/env python
# coding: utf-8

# In[1]:


name="tayyab"
salary=300000
print(f"{name} is earning total of {salary},every month" )


# In[1]:


name="tayyab"
salary=300000
print(f"{name} is earning total of {salary},every month" )


# In[2]:


from sklearn import datasets


# In[3]:


iris_datasets=datasets.load_iris()


# In[4]:


x=iris_datasets.data[:,2]


# In[13]:


count=x.(len.flat)


# In[14]:


import pandas as pd 


# In[15]:


import numpy as np 


# In[16]:


import seaborn as sns 


# In[4]:


from sklearn.model_selection import train_test_split   
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt
import seaborn as sns   
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


# In[18]:


pd.read_csv("from sklearn.model_selection import train_test_split   
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt
import seaborn as sns   
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np")


# In[20]:


import sys 
print('pyton:{}'.format(sys.version ))


# In[6]:


#scipy is a powerful open-source Python library used for scientific and technical computing
import scipy
print('scipy:{}'.format(scipy.__version__))


# In[7]:


import numpy
print('numpy:{}'.format(numpy.__version__))


# In[3]:


import sklearn
print('sklearn:{}'.format(sklearn.__version__))


# In[8]:


import pandas as pd 
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt #pyplot is a drawing tool that is use for charts and graphs 

from sklearn import model_selection #model_selection is a module to split data in training and testing data 

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix #confusion matrix see where you algorithm is right how much prediction is right vs wrong 

from sklearn.metrics import accuracy_score #accuracy tells you the percentage of  correct predictions your model has made

from sklearn.linear_model import LogisticRegression# binary classification is used for binary classififcations

from sklearn.tree import DecisionTreeClassifier# it is used to make tree like structure usually used to make predictions 



# In[2]:


import pandas as pd 


# In[3]:


df1=pd.read_csv(r"C:\Users\HP\AppData\Local\Temp\Rar$DIa15140.47283.rartemp\Iris.csv")


# In[1]:


import pandas as pd 


# In[2]:


df1=pd.read_csv(r"C:\Users\HP\AppData\Local\Temp\Rar$DIa23008.25127.rartemp\Iris.csv")
df1


# In[6]:


df1.tail()


# In[10]:


import seaborn as sns 
sns.boxplot(x=df1["SepalWidthCm"],y=df1["PetalWidthCm"],color='green')
sns.title("box plot")
plt.show()


# In[9]:


df1.shape


# In[12]:


df1.isnull()


# In[15]:


df1.describe()


# In[17]:


df1.groupby('class').size()


# In[19]:


df1


# In[27]:


df1.groupby('class').size()


# In[37]:


df1.plot(kind="box", subplots=True, layout=(2, 3), sharex=False, sharey=True)



# In[41]:


df1.hist()
plt.show()


# In[40]:


import matplotlib.pyplot as plt 


# In[45]:


scatter_matrix(df1)
plt.show()


# In[44]:


from pandas.plotting import scatter_matrix


# In[3]:


from sklearn import model_selection

array = df1.values
x = array[:, 0:4]           # Features (first 4 columns)
y = array[:, 4]             # Target (Species)

validation_size = 0.20      # 20% test data
seed = 6                    # For reproducibility

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=validation_size, random_state=seed
)


# In[4]:


from sklearn import model_selection 
#this imports the model_selection form sklearn library which divides the datasets 


# In[5]:


now we are going to create a test harness we will use 10 fold-cross validation it will test the accuracy train on the 9 parts and test on 1 part 


# In[6]:


seed=6
scoring="accuracy "


# In[9]:


models = []
models.append(('LR', LogisticRegression(max_iter=200)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluate each model using cross-validation
results = []
names = []

print("Model Accuracy Results (10-fold cross-validation):\n")
for name, model in models:
    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: Mean Accuracy = {cv_results.mean():.3f}, Std = {cv_results.std():.3f}")


# In[13]:


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load your data
# df1 = pd.read_csv('your_file.csv')  # Make sure df1 is defined

# Extract features and target
X = df1.iloc[:, 0:3].values
y = df1.iloc[:, 2].values

# Encode target labels if they are strings
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Cross-validation setup
seed = 6
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)

# Models to compare
models = []
models.append(('LR', LogisticRegression(max_iter=200)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluate each model
print("Model Accuracy Results (10-fold cross-validation):\n")
for name, model in models:
    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    print(f"{name}: Mean Accuracy = {cv_results.mean():.3f}, Std = {cv_results.std():.3f}")


# In[12]:


df1


# In[39]:


models=[]
models.append(('Lr',LogisticRegression(max_iter=2000)))
models.append(('KNm',KNeighborsClassifier ()))
models.append(('svm',SVC()))
models.append(('DT',DecisionTreeClassifier()))
models.append(("NB",GaussianNB()))


# In[19]:


from sklearn.linear_model import LogisticRegressionmode


# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# In[56]:


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


X = df1.iloc[:, 0:3].values
y = df1.iloc[:, 2].values

# Encode target labels if they are strings
encoder = LabelEncoder()
y = encoder.fit_transform(y)
seed=6
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
models=[]
models.append(("LR",LogisticRegression(max_iter=2000)))
models.append(('KNN',KNeighborsClassifier()))
models.append(("DT",DecisionTreeClassifier()))
models.append(("svm",SVC()))
models.append(('NB',GaussianNB()))
print("Model accuracy results (10-fold cross -validatuion)")
for name,model in models:
    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    print(f"{name}: Mean Accuracy = {cv_results.mean():.3f}, Std = {cv_results.std():.3f}")


# In[81]:


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
x=df1.iloc[:,0:3].values
y=df1.iloc[:,4].values

encoder=LabelEncoder()
y=encoder.fit_transform(y)

models=[]
models.append(('LR',LogisticRegression(max_iter=2000)))
models.append(('svm',SVC()))
models.append(('DT',DecisionTreeClassifier()))
models.append(('kNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
print("model accuracy results(10-fold cross-validation)")
for name,model in models:
    csv_results=modek


# In[ ]:




