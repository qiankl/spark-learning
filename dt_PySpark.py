#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd


# In[2]:


from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
conf = SparkConf().setAppName("getdata").setMaster("yarn")
sc = SparkContext(conf=conf)
# sc= SparkContext()
sqlContext = SQLContext(sc)

CV_data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('churn-bigml-80.csv')


# In[3]:


test_data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('churn-bigml-20.csv')


# In[4]:


CV_data.cache()
CV_data.printSchema()


# In[5]:


pd.DataFrame(CV_data.take(5), columns = CV_data.columns).transpose()


# In[6]:


CV_data.describe().toPandas().transpose()


# In[7]:


CV_data.dtypes


# In[8]:


numeric_features = [t[0] for t in CV_data.dtypes if t[1] == 'int' or t[1] == 'double']


# In[9]:


sampled_data = CV_data.select(numeric_features).sample(False, 0.1).toPandas()


# In[13]:


axs = pd.plotting.scatter_matrix(sampled_data, figsize=(12, 12))
n = len(sampled_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())


# In[14]:


from pyspark.sql.types import DoubleType
from pyspark.sql.functions import UserDefinedFunction

CV_data = CV_data.drop("State").drop("Area code")                        .drop("Total day charge").drop("Total eve charge")                        .drop("Total night charge").drop("Total intl charge")                        .withColumn("Churn",
                                   CV_data["Churn"].cast(DoubleType())) \
                       .withColumn("International plan",
                                   CV_data["International plan"]
                                   .cast("boolean").cast(DoubleType())) \
                       .withColumn("Voice mail plan",
                                   CV_data["Voice mail plan"]
                                   .cast("boolean").cast(DoubleType()))
test_data = test_data.drop("State").drop("Area code")                        .drop("Total day charge").drop("Total eve charge")                        .drop("Total night charge").drop("Total intl charge")                        .withColumn("Churn",
                                   test_data["Churn"].cast(DoubleType())) \
                       .withColumn("International plan",
                                   test_data["International plan"]
                                   .cast("boolean").cast(DoubleType())) \
                       .withColumn("Voice mail plan",
                                   test_data["Voice mail plan"]
                                   .cast("boolean").cast(DoubleType()))


# In[15]:


pd.DataFrame(CV_data.take(5), columns = CV_data.columns).transpose()


# In[16]:


from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# In[17]:


vecAssembler = VectorAssembler(inputCols = ['Account length', 'International plan', 'Voice mail plan', 'Number vmail messages', 
                                           'Total day minutes', 'Total day calls', 'Total eve minutes', 'Total eve calls', 
                                          'Total night minutes', 'Total night calls', 'Total intl minutes', 'Total intl calls', 
                                          'Customer service calls'], outputCol = 'features')


# In[18]:


df_train = vecAssembler.transform(CV_data)


# In[19]:


pd.DataFrame(df_train.take(5), columns = df_train.columns).transpose()


# In[20]:


from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol = 'Churn', featuresCol = 'features')


# In[21]:


pipeline = Pipeline(stages=[vecAssembler, dt])


# In[22]:


model = pipeline.fit(CV_data)


# In[23]:


predictions = model.transform(test_data)


# In[24]:


predictions.select('prediction', 'Churn', 'features').toPandas().head(20)


# In[25]:


evaluator = BinaryClassificationEvaluator(labelCol = 'Churn', rawPredictionCol='prediction')


# In[26]:


evaluator.evaluate(predictions)


# In[27]:


paramGrid = ParamGridBuilder().addGrid(dt.maxDepth, [2,3,4,5,6,7]).build()

# Set up 3-fold cross validation
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator, 
                          numFolds=3)

CV_model = crossval.fit(CV_data)


# In[28]:


tree_model = CV_model.bestModel.stages[1]
print(tree_model)


# In[29]:


predictions_improved = CV_model.bestModel.transform(test_data)


# In[30]:


predictions_improved.select('prediction', 'Churn', 'features').toPandas().head(20)


# In[31]:


evaluator.evaluate(predictions_improved)


# In[ ]:




