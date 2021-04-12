#!/usr/bin/env python
# coding: utf-8

# #### SANDRA MARIA JOSEPH 2048051

# # <font color=red>*CLASSIFICATION OF YOUTUBE VIDEOS*</font>:

# ![image.png](attachment:image.png)

# ### <font color=blue>Data Gathering and Preprocessing</font>

# In[1]:


import numpy as np
import pandas as pd
import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ### *Importing scikit-learn classifiers*

# In[2]:


#importing Multinomial Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

#importing Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

#importing Support Vector Classifier
from sklearn.svm import SVC

#importing Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier


# ### Importing Data

# In[3]:


data_video = pd.read_csv(r"E:\MDS_SEM2\New folder\Machine_Learning\Lab\program_2\dAta\USvideos.csv")
data_video.head(5)


# ### <font color=blue>Deleting The unwanted columns and Generating the new csv file</font>

# In[4]:


fresh_columns = ['title', 'category_id']
fresh_video = data_video[fresh_columns]
fresh_video.to_csv(r"E:\MDS_SEM2\New folder\Machine_Learning\Lab\program_2\dAta\USvideos22.csv", index=False)
fresh_video = pd.read_csv(r"E:\MDS_SEM2\New folder\Machine_Learning\Lab\program_2\dAta\USvideos22.csv", header=0, names=['Title', 'Category_ID'])
fresh_video


# ### <font color=blue>Importing JSON file</font>

# In[5]:


category_json = pd.read_json(r'E:\MDS_SEM2\New folder\Machine_Learning\Lab\program_2\dAta\US_category_id.json')
category_json.head(5)


# ### <font color=green>Creating a list of Dictionaries with ID and Category label mapping</font>

# In[6]:


category_dict = [{'id': item['id'], 'title': item['snippet']['title']} for item in category_json['items']]
category_dict


# ### <font color=green>Creating a DataFrame for the Dictionary</font>

# In[7]:


category_df = pd.DataFrame(category_dict)
category_df.head(5)


# In[8]:


categories = category_df.rename(index=str, columns = {"id":"Category_ID","title":"Category"})
categories.head(5)


# ### <FONT COLOR=RED>Feature Extraction (or Vectorization).</font>
# In order to use textual data for predictive modeling, the text must be parsed to remove certain words – this process is called tokenization. These words need to then be encoded as integers, or floating-point values, for use as inputs in machine learning algorithms. This process is called feature extraction (or vectorization).

# *Scikit-learn’s CountVectorizer is used to convert a collection of text documents to a vector of term/token counts. It also enables the ​pre-processing of text data prior to generating the vector representation. This functionality makes it a highly flexible feature representation module for text.*

# In[9]:


vector = CountVectorizer()


# **CountVectorizer() takes in a corpus with multiple documents(bunch of text with multiple statements in layman’s terms) as input.**
# 
# * When you apply countvectorizer.fit on corpus1 it removes all the stopwords (words which occur frequently in every sentence and are worthless for our algorithm/system. ex: is,the,okay) and then it tokenizes the total text (gives distinct number to distinct word of whole text corpus), all the distinct words which got distinct indices(number) constitute together to form vocabulary.
# 
# * when you apply countvectorizer.fit, countvectorizer maps every document(sentence) to vocabulary(tokens) where documents are rows and tokens are along columns
# 
# 
# *from sklearn.feature_extraction.text import CountVectorizer
# 
# * *CountVectorizer.fit(data) #Learn the vocabulary of the training data*
# 
# * *CountVectorizer.transform(data) #Converts the training data into the Document Term Matrix*

# In[10]:


counts = vector.fit_transform(fresh_video['Title'].values)
counts


# ## Using various classification models and targetting 'Category'

# * Naive Bayes classifier for multinomial models.The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.

# In[11]:


NB_Model = MultinomialNB()


# In[12]:


###A random forest classifier.
'''A random forest is a meta estimator that fits a number of decision tree classifiers 
on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.'''

RFC_Model = RandomForestClassifier()

##C-Support Vector Classification.
'''The implementation is based on libsvm. The fit time scales at least quadratically with the number 
of samples and may be impractical beyond tens of thousands of samples.'''

SVC_Model = SVC()

##Decision Tree Classifier
'''Dcision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. 
The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred 
from the data features. A tree can be seen as a piecewise constant approximation.'''

DTC_Model = DecisionTreeClassifier()


# In[13]:


output = fresh_video['Category_ID'].values


# In[14]:


NB_Model.fit(counts,output)


# In[15]:


RFC_Model.fit(counts,output)


# In[16]:


SVC_Model.fit(counts,output)


# In[17]:


DTC_Model.fit(counts,output)


# ## Checking the accuracy using 90/10 train/test split

# In[18]:


X = counts
Y = output
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .1)


# In[19]:


NBtest = MultinomialNB().fit(X_train,Y_train)
nb_predictions = NBtest.predict(X_test)
acc_nb = NBtest.score(X_test, Y_test)
print('The Naive Bayes Algorithm has an accuracy of', acc_nb)


# In[20]:


RFCtest = RandomForestClassifier().fit(X_train,Y_train)
rfc_predictions = RFCtest.predict(X_test)
acc_rfc = RFCtest.score(X_test, Y_test)
print('The Random Forest Algorithm has an accuracy of', acc_rfc)


# In[21]:


SVCtest = SVC().fit(X_train,Y_train)
svc_predictions = SVCtest.predict(X_test)
acc_svc = SVCtest.score(X_test, Y_test)
print('The Support Vector Algorithm has an accuracy of', acc_svc)


# In[22]:


DTCtest = DecisionTreeClassifier().fit(X_train,Y_train)
dtc_predictions = DTCtest.predict(X_test)
acc_dtc = DTCtest.score(X_test, Y_test)
print('The Decision Tree Algorithm has an accuracy of', acc_dtc)


# ## Entering titles to Predict the Category

# In[23]:


Titles = ["Liverpool vs Barcelona football match highlights"]


# In[24]:


##Inserting above titles into each classifier model


# In[25]:


Titles_counts = vector.transform(Titles)


# In[26]:


##NAIVE BAYES MODEL

PredictNB = NB_Model.predict(Titles_counts)
PredictNB


# In[27]:


#RANDOM FOREST MODEL

PredictRFC = RFC_Model.predict(Titles_counts)
PredictRFC


# In[28]:


#SUPPORT VECTOR CLASSIFIER MODEL

PredictSVC = SVC_Model.predict(Titles_counts)
PredictSVC


# In[29]:


#DECISION TREE MODEL

PredictDTC = DTC_Model.predict(Titles_counts)
PredictDTC


# ## <FONT COLOR=BLUE>Output will be an array of numbers. Iterate through the Category Dictionary (from JSON file) to find "title"</FONT>

# In[30]:


CategoryNamesListNB = []
for Category_ID in PredictNB:
    MatchingCategoriesNB = [x for x in category_dict if x["id"] == str(Category_ID)]
    if MatchingCategoriesNB:
        CategoryNamesListNB.append(MatchingCategoriesNB[0]["title"])


# In[31]:


CategoryNamesListRFC = []
for Category_ID in PredictRFC:
    MatchingCategoriesRFC = [x for x in category_dict if x["id"] == str(Category_ID)]
    if MatchingCategoriesRFC:
        CategoryNamesListRFC.append(MatchingCategoriesRFC[0]["title"])


# In[32]:


CategoryNamesListSVC = []
for Category_ID in PredictSVC:
    MatchingCategoriesSVC = [x for x in category_dict if x["id"] == str(Category_ID)]
    if MatchingCategoriesSVC:
        CategoryNamesListSVC.append(MatchingCategoriesSVC[0]["title"])


# In[33]:


CategoryNamesListDTC = []
for Category_ID in PredictDTC:
    MatchingCategoriesDTC = [x for x in category_dict if x["id"] == str(Category_ID)]
    if MatchingCategoriesDTC:
        CategoryNamesListDTC.append(MatchingCategoriesDTC[0]["title"])


# ## <FONT COLOR=BLUE>Mapping these values to the Titles we want to Predict</FONT>

# In[34]:


TitleDataFrameNB = []
for i in range(0, len(Titles)):
    TitleToCategoriesNB = {'Title': Titles[i],  'Category': CategoryNamesListNB[i]}
    TitleDataFrameNB.append(TitleToCategoriesNB)


# In[35]:


TitleDataFrameRFC = []
for i in range(0, len(Titles)):
    TitleToCategoriesRFC = {'Title': Titles[i],  'Category': CategoryNamesListRFC[i]}
    TitleDataFrameRFC.append(TitleToCategoriesRFC)


# In[36]:


TitleDataFrameSVC = []
for i in range(0, len(Titles)):
    TitleToCategoriesSVC = {'Title': Titles[i],  'Category': CategoryNamesListSVC[i]}
    TitleDataFrameSVC.append(TitleToCategoriesSVC)


# In[37]:


TitleDataFrameDTC = []
for i in range(0, len(Titles)):
    TitleToCategoriesDTC = {'Title': Titles[i],  'Category': CategoryNamesListDTC[i]}
    TitleDataFrameDTC.append(TitleToCategoriesDTC)


# ## Converting the resulting Dictionary to a Data Frame

# In[38]:


PredictDFnb = pd.DataFrame(PredictNB)
TitleDFnb = pd.DataFrame(TitleDataFrameNB)
PreFinalDFnb = pd.concat([PredictDFnb, TitleDFnb], axis=1)
PreFinalDFnb.columns = (['Categ_ID', 'Predicted Category', 'Hypothetical Video Title'])
FinalDFnb = PreFinalDFnb.drop(['Categ_ID'],axis=1)
colsNB = FinalDFnb.columns.tolist()
colsNB = colsNB[-1:] + colsNB[:-1]
FinalDFnb= FinalDFnb[colsNB]


# In[39]:


PredictDFrfc = pd.DataFrame(PredictRFC)
TitleDFrfc = pd.DataFrame(TitleDataFrameRFC)
PreFinalDFrfc = pd.concat([PredictDFrfc, TitleDFrfc], axis=1)
PreFinalDFrfc.columns = (['Categ_ID', 'Predicted Category', 'Hypothetical Video Title'])
FinalDFrfc = PreFinalDFrfc.drop(['Categ_ID'],axis=1)
colsRFC = FinalDFrfc.columns.tolist()
colsRFC = colsRFC[-1:] + colsRFC[:-1]
FinalDFrfc= FinalDFrfc[colsRFC]


# In[40]:


PredictDFsvc = pd.DataFrame(PredictSVC)
TitleDFsvc = pd.DataFrame(TitleDataFrameSVC)
PreFinalDFsvc = pd.concat([PredictDFsvc, TitleDFsvc], axis=1)
PreFinalDFsvc.columns = (['Categ_ID', 'Predicted Category', 'Hypothetical Video Title'])
FinalDFsvc = PreFinalDFsvc.drop(['Categ_ID'],axis=1)
colsSVC = FinalDFsvc.columns.tolist()
colsSVC = colsSVC[-1:] + colsSVC[:-1]
FinalDFsvc= FinalDFsvc[colsSVC]


# In[41]:


PredictDFdtc = pd.DataFrame(PredictDTC)
TitleDFdtc = pd.DataFrame(TitleDataFrameDTC)
PreFinalDFdtc = pd.concat([PredictDFdtc, TitleDFdtc], axis=1)
PreFinalDFdtc.columns = (['Categ_ID', 'Predicted Category', 'Hypothetical Video Title'])
FinalDFdtc = PreFinalDFdtc.drop(['Categ_ID'],axis=1)
colsDTC = FinalDFdtc.columns.tolist()
colsDTC = colsDTC[-1:] + colsDTC[:-1]
FinalDFdtc= FinalDFdtc[colsDTC]


# # Final Prediction

# In[44]:


Titles = ["How Black Panther Should Have Ended"]


# In[45]:


Titles_counts = vector.transform(Titles)
PredictDTC = DTC_Model.predict(Titles_counts)

CategoryNamesListDTC = []
for Category_ID in PredictDTC:
    MatchingCategoriesDTC = [x for x in category_dict if x["id"] == str(Category_ID)]
    if MatchingCategoriesDTC:
        CategoryNamesListDTC.append(MatchingCategoriesDTC[0]["title"])

TitleDataFrameDTC = []
for i in range(0, len(Titles)):
    TitleToCategoriesDTC = {'Title': Titles[i],  'Category': CategoryNamesListDTC[i]}
    TitleDataFrameDTC.append(TitleToCategoriesDTC)
    
PredictDFdtc = pd.DataFrame(PredictDTC)
TitleDFdtc = pd.DataFrame(TitleDataFrameDTC)
PreFinalDFdtc = pd.concat([PredictDFdtc, TitleDFdtc], axis=1)
PreFinalDFdtc.columns = (['Categ_ID', 'Hypothetical Video Title','Predicted Category'])
FinalDFdtc = PreFinalDFdtc.drop(['Categ_ID'],axis=1)
colsDTC = FinalDFdtc.columns.tolist()
colsDTC = colsDTC[-1:] + colsDTC[:-1]
FinalDFdtc= FinalDFdtc[colsDTC]

# Decision Trees
FinalDFdtc


# The given tiltle of the video is 'How Black Panther Should Have Ended'. It's category is clearly identified as the film and Animation. 
