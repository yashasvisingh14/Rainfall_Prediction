
import pandas as pd 
import matplotlib.pyplot as plt
df=pd.read_csv("C:/Users/HP/Documents/weatherHistory.csv")
print(df.head())
df['RAIN'].unique()
df[df['RAIN'].isnull()]
df=df.dropna()
df.shape
df['RAIN']=df['RAIN'].astype('int')
df['RAIN'].value_counts()
df=df.drop('DATE',axis=1)
import seaborn as sns
plt.figure(figsize=(8, 8))
sns.heatmap(df.corr(),annot=True)
y=df['RAIN']
X=df.drop(['RAIN'],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(X_train,y_train)
pred=clf.predict(X_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,pred))
from sklearn import metrics
print(metrics.classification_report(y_test, pred))
clf.feature_importances_