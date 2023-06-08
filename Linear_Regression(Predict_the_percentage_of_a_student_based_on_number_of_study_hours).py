#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression with Python Scikit Learn
# ##### In this section we will see how the Python Scikit-Learn library for machine learning can be used to implement regression functions. We will start with simple linear regression involving two variables.

# ## Simple Linear Regression
# #### In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# In[1]:
# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:
# Reading data from remote link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")
s_data.head(10)


# #### Let's plot our data points on 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data. We can create the plot with the following script:
# In[3]:
# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ## Preparing the data
# #### The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).
# In[4]:
X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  


# In[5]:
X


# In[6]:
y


# #### Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:
# In[7]:
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# In[8]:
X_train


# In[9]:
X_test


# In[10]:
y_train


# In[11]:
y_test


# ## Training the Algorithm
# #### We have split our data into training and testing sets, and now is finally the time to train our algorithm.
# In[12]:
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 
print("Training complete.")


# In[13]:
# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_
# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# ## Making Predictions
# #### Now that we have trained our algorithm, it's time to make some predictions.
# In[14]:
print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[15]:
# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[16]:
import numpy as np
# You can also test with your own data
hours = 9.25
own_pred = regressor.predict(np.array(hours).reshape(-1, 1))
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# ## Evaluating the model
# #### The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.
# In[17]:
from sklearn import metrics  
print('Mean Squared Error:', 
      metrics.mean_squared_error(y_test, y_pred)) 


# ## Examples of other metrics
# #### 1. RMSE(ROOT MEAN SQUARE ERROR)
# In[18]:
import numpy as np
from sklearn import metrics
print('Root Mean Square Error:', 
      np.sqrt(metrics.mean_squared_error(y_test, y_pred)) )


# #### 2. MAE(Mean Absolute Error)
# In[19]:
from sklearn import metrics
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# #### 3. R-Squared(R2) Score 
# In[20]:
from sklearn import metrics
print('R-Squared(R2) Score:', 
      metrics.r2_score(y_test, y_pred),'- The R2 score ranges from 0 to 1, where 1 indicates a perfect fit and 0 indicates that the model does not explain any of the variance in the target variable. Higher values of the R2 score indicate better performance.') 




