import pandas as pd 
def preprocess_data(df):
   #dropping id column and converting categorical column to numeric
   df=df.drop(columns=["transaction_id"])
   df=pd.get_dummies(df,columns=["merchant_category"])
   return df