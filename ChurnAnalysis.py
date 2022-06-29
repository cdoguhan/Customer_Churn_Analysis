
import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing

#sonuçların yeniden üretilebilir olmasını amaçlıyoruz
np.random.seed(1)

columns = [
     'state',
     'account length', 
     'area code', 
     'phone number', 
     'international plan', 
     'voice mail plan', 
     'number vmail messages',
     'total day minutes',
     'total day calls',
     'total day charge',
     'total eve minutes',
     'total eve calls',
     'total eve charge',
     'total night minutes',
     'total night calls',
     'total night charge',
     'total intl minutes',
     'total intl calls',
     'total intl charge',
     'number customer service calls',
     'churn']

#########
data = pd.read_csv('ChurnDataset.txt', header = None, names = columns)
#Datasetin orjinali hali
print("Dataset orjinal hali: " + str(data.shape))


data.head(10)

data.dtypes


data.shape

pd.DataFrame(data.isnull().sum(),columns=["Count"])

###########
corr_1 = np.abs(data[data.columns[2]].corr(data[data.columns[20]]
                                           ,method = "spearman"))

corr_2 = np.abs(data[data.columns[3]].corr(data[data.columns[20]]
                                           ,method = "spearman"))



corr_1

corr_2

data.info


data.describe().T

data.describe()

from scipy.stats import stats
stats.spearmanr(data["total day minutes"],data["churn"])


corr_1 = np.abs(data[data.columns[1]].corr(data[data.columns[20]]
                                           ,method = "spearman"))


corr_2 = np.abs(data[data.columns[6]].corr(data[data.columns[20]]
                                           ,method = "spearman"))


corr_3 = np.abs(data[data.columns[7]].corr(data[data.columns[20]]
                                           ,method = "spearman"))

corr_4 = np.abs(data[data.columns[8]].corr(data[data.columns[20]]
                                           ,method = "spearman"))

corr_5 = np.abs(data[data.columns[9]].corr(data[data.columns[20]]
                                           ,method = "spearman"))

corr_6 = np.abs(data[data.columns[10]].corr(data[data.columns[20]]
                                           ,method = "spearman"))


corr_7 = np.abs(data[data.columns[11]].corr(data[data.columns[20]]
                                           ,method = "spearman"))


corr_8 = np.abs(data[data.columns[12]].corr(data[data.columns[20]]
                                           ,method = "spearman"))


corr_9 = np.abs(data[data.columns[13]].corr(data[data.columns[20]]
                                           ,method = "spearman"))


corr_10 = np.abs(data[data.columns[14]].corr(data[data.columns[20]]
                                           ,method = "spearman"))


corr_11 = np.abs(data[data.columns[15]].corr(data[data.columns[20]]
                                           ,method = "spearman"))


corr_12 = np.abs(data[data.columns[16]].corr(data[data.columns[20]]
                                           ,method = "spearman"))


corr_13 = np.abs(data[data.columns[17]].corr(data[data.columns[20]]
                                           ,method = "spearman"))


corr_14 = np.abs(data[data.columns[18]].corr(data[data.columns[20]]
                                           ,method = "spearman"))


corr_15 = np.abs(data[data.columns[19]].corr(data[data.columns[20]]
                                           ,method = "spearman"))



corr_1  # account length ile churn arasındaki

corr_2 # number v mail messages ile churn arasındaki

corr_3 # total day minutes ile churn arasındaki

corr_4 # total day calls ile churn arasındaki

corr_5 # total day charge ile churn arasındaki

corr_6 # total eve minutes ile churn arasındaki

corr_7 #total eve calls ile churn arasındaki

corr_8 #total eve charge ile churn arasındaki

corr_9 #total night minutes ile churn arasındaki

corr_10 #total night calls ile churn arasındaki

corr_11 #total night charge ile churn arasındaki

corr_12 #total intl minutes ile churn arasındaki

corr_13 #total intl call ile churn arasındaki

corr_14 #total intl charge ile churn arasındaki

corr_15 #number customer service calls ile churn arasındaki

#############
import seaborn as sns
sns.boxplot(x=data["account length"])

sns.boxplot(x=data["number vmail messages"])

sns.boxplot(x=data["total day minutes"])

sns.boxplot(x=data['total day calls'])

sns.boxplot(x=data["total day charge"])

sns.boxplot(x=data["total eve minutes"])

sns.boxplot(x=data["total eve calls"])

sns.boxplot(x=data["total eve charge"])

sns.boxplot(x=data['total night calls'])

###########
data.head(10)

data["account length"].mean()

data.drop('phone number', axis = 1, inplace = True)
data.drop('area code', axis = 1, inplace = True)
data.drop('state', axis = 1, inplace = True)
data.drop('total night calls',axis=1 , inplace=True)
print("Dataset preprocessing sonrasi: " + str(data.shape))


data1 = data.select_dtypes(include=["float64","int64"])
df=data1.copy()
df=df.dropna()
df.head()

##################################################
from sklearn.neighbors import LocalOutlierFactor

clf=LocalOutlierFactor(n_neighbors=20,contamination=0.1)

clf.fit_predict(df)

df_scores=clf.negative_outlier_factor_

np.sort(df_scores)[0:50]

esik_deger=np.sort(df_scores)[12]

aykiri_tf=df_scores>esik_deger
aykiri_tf

data5=data[df_scores>esik_deger]

data5
data6=data5.copy()
#############################################
from sklearn.preprocessing import LabelEncoder
lbe=LabelEncoder()


lbe.fit_transform(data6["international plan"])

data6["international plan"]=lbe.fit_transform(data6["international plan"])

lbe.fit_transform(data6["voice mail plan"])

data6["voice mail plan"]=lbe.fit_transform(data6["voice mail plan"])

lbe.fit_transform(data6["churn"])

data6["churn"]=lbe.fit_transform(data6["churn"])

data7=data6.copy()

data10 = data7[data7['churn']==1]
print("Churn olanlar-data10:"+ str(data10.shape))

data11 = data7[data7['churn']==0]
print("Churn olmayanlar-data11:"+ str(data11.shape))

data7 = data10.append(data11[:481])
print("Son veriseti :"+ str(data7.shape))

#################################################
from sklearn import preprocessing

#Egitim  ve test verisini parcaliyoruz --> 80% / 20%
X = data7.loc[:, data7.columns != 'churn']
Y = data7['churn']
X_train, X_test, Y_train, Y_test =model_selection.train_test_split(X, Y, test_size=0.2, random_state=0)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#ölçeklendirme
scaler = preprocessing.MinMaxScaler((-1,1))
scaler.fit(X)
XX_train = scaler.transform(X_train.values)
XX_test  = scaler.transform(X_test.values)
YY_train = Y_train.values 
YY_test  = Y_test.values

#########################################
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
import matplotlib.pyplot as plt
import scikitplot.metrics as splt
import sklearn.metrics as mt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Decision Tree (CART)',DecisionTreeClassifier())) 
models.append(('K-NN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('AdaBoostClassifier', AdaBoostClassifier()))
models.append(('BaggingClassifier', BaggingClassifier()))
models.append(('RandomForestClassifier', RandomForestClassifier()))


for name, model in models:
    model = model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    from sklearn import metrics
    print("Model -> %s -> ACC: %%%.2f" % (name,metrics.accuracy_score(Y_test, Y_pred)*100))

####################################################

get_ipython().system('pip install scikit-plot')

report = classification_report(YY_test, Y_pred)
print(report)


rf = RandomForestClassifier()

rf.fit(X_train,Y_train)


ranking=rf.feature_importances_
features=np.argsort(ranking)[::-1][:17]
columns=X.columns


plt.title("Random Forest Classifier modeline göre degiskenlerin önem derecesi", y = 1.03, size = 18)
plt.bar(range(len(features)), ranking[features], color="aqua", align="center")
plt.xticks(range(len(features)), columns[features], rotation=80)
plt.show()

rf.predict_proba(X_test)
###################################