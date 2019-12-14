# Home_Sale_Price_Prediction_Kaggle-Competition

import pandas as pd
import numpy as np
import seaborn as sns

#load the data set in Jupyter
#Explotary data Analysis

a  = pd.read_csv(r"C:\akshay\House_price_Prediction Kaggle\train.csv")
b = pd.read_csv(r"C:\akshay\House_price_Prediction Kaggle\test.csv")

print("The shape of train data",a.shape[0], "Houses" ,"and" ,a.shape[1], "Features")
print("The shape of test data", b.shape[0], "Houses", "and" ,b.shape[1], "Feature")

#look first at the correlation between numerical features and the target "SalePrice", in order to have a first idea of the connections between features.

import matplotlib.pyplot as plt
#%matplotlib inline
#from scipy.stats import norm
#from sklearn.preprocessing import StandardScaler
from scipy import stats
num = a.select_dtypes(exclude = "object")
numcorr = num.corr()
f,ax=plt.subplots(figsize=(17,1))
sns.heatmap(numcorr.sort_values(by = ["SalePrice"], ascending=False).head(1),cmap = "Blues")
plt.title("Corelation between numerical feature and Sale Price", weight = "bold" , fontsize  = 18)
plt.show()
#looking at the heatmap above we can see many dark colors, many features have high correlation with the target.

Num = numcorr["SalePrice"].sort_values(ascending= False).head(10).to_frame()
cm = sns.light_palette("cyan", as_cmap=True)
s = Num.style.background_gradient(cmap = cm)
s

#we have finding the variables having more correlation between the sales price and numberical variable
# from the below table we have conclude that overall quality,living area,garage area,garage car are highly corelated with sales price


Missing value Treatment
plt.style.use("seaborn")
sns.set_style("whitegrid")
plt.subplots(0,0,figsize=(15,3))
a.isnull().mean().sort_values(ascending = False).plot.bar(color = "black")
plt.axhline(y =0.1,color = "red" ,linestyle = "-")
plt.title("MISSING VALUES AVRAGE PER COLUMN: TRAIN DATA SET", Weight = "bold")
plt.show()

plt.subplots(0,0,figsize=(15,3))
b.isnull().mean().sort_values(ascending = False).plot.bar(color = "black")
plt.axhline(y =0.1,color = "red" ,linestyle = "-")
plt.show()


na = a.shape[0]
nb =  b.shape[0]

y_a = a["SalePrice"].to_frame()
y_a

#combine Test and train data

T = pd.concat((a,b), sort = False).reset_index(drop = True)

#drop taget and id variable from data

T.drop(["SalePrice"],axis = 1, inplace = True)
T.drop(["Id"],axis = 1, inplace = True)

print("Total size of :",T.shape)


T1 = T.dropna(thresh = len(T)*0.9, axis =1) 
print("we have drop:",T.shape[1]-T1.shape[1],"Feature from the combbined dataset")

allna = (T.isnull().sum() / len(T))
allna = allna.drop(allna[allna == 0].index).sort_values(ascending=False)
plt.figure(figsize=(12, 8))
allna.plot.barh(color='purple')
plt.title('Missing values average per column', fontsize=25, weight='bold' )
plt.show()

print('The shape of the combined dataset after dropping features with more than 90% M.V.', T.shape)

NA=T[['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','GarageYrBlt','BsmtFinType2','BsmtFinType1','BsmtCond', 'BsmtQual','BsmtExposure', 'MasVnrArea','MasVnrType','Electrical','MSZoning','BsmtFullBath','BsmtHalfBath','Utilities','Functional','Exterior1st','BsmtUnfSF','Exterior2nd','TotalBsmtSF','GarageArea','GarageCars','KitchenQual','BsmtFinSF2','BsmtFinSF1','SaleType']]


#We are splitting cat and numerical variable from data set

Tcat = NA.select_dtypes(include = "object")
Tnum = NA.select_dtypes(exclude = "object" )

print("we have",Tcat.shape, "categorical variable")
print("we have",Tnum.shape, "numerical variable")


Tnum.isnull().sum().sort_values(ascending = False).head(3)

#T["LotFrontage"] = T.LotFrontage.fillna(0)
T["GarageYrBlt"] = T.GarageYrBlt.fillna(1980)
T["MasVnrArea"] = T.MasVnrArea.fillna(0)


Tcat.head()

Tcat.isnull().sum().sort_values(ascending = False)


#We start with features having just few missing value:  We fill the gap with forward fill method:
T['Electrical']=T['Electrical'].fillna(method='ffill')
T['SaleType']=T['SaleType'].fillna(method='ffill')
T['KitchenQual']=T['KitchenQual'].fillna(method='ffill')
T['Exterior1st']=T['Exterior1st'].fillna(method='ffill')
T['Exterior2nd']=T['Exterior2nd'].fillna(method='ffill')
T['Functional']=T['Functional'].fillna(method='ffill')
T['Utilities']=T['Utilities'].fillna(method='ffill')
T['MSZoning']=T['MSZoning'].fillna(method='ffill')



#Categorical missing values
NAcols=T.columns
for col in NAcols:
    if T[col].dtype == "object":
        T[col] = T[col].fillna("None")
        
        
        #Numerical missing values
for col in NAcols:
    if T[col].dtype != "object":
        T[col]= T[col].fillna(0)
        
        
T.isnull().sum().sort_values(ascending = False).head(2)


##LABEL ENCODING BLOCK

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

for x in Tcat:
    T[x]=le.fit_transform(T[x])
    
T.head()


Train = T[:na]
test = T[na:]

print("shape of trainig data is:",Train.shape)
print("shape of testing data is:",test.shape)


#Detecting Outlier

fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(16, 4))
axes = np.ravel(axes)

col_name = ['GrLivArea','TotalBsmtSF','1stFlrSF','BsmtFinSF1','LotArea']

for i, c in zip(range(5), col_name):
    a.plot.scatter(ax=axes[i], x=c, y='SalePrice', sharey=True, colorbar=False, c='r')
    
    # delete outliers
print(a.shape)
a = a[a['GrLivArea'] < 4500]
a = a[a['LotArea'] < 100000]
a = a[a['TotalBsmtSF'] < 3000]
a = a[a['1stFlrSF'] < 2500]
a = a[a['BsmtFinSF1'] < 2000]

print(a.shape)


for i, c in zip(range(5,10), col_name):
    a.plot.scatter(ax=axes[i], x=c, y='SalePrice', sharey=True, colorbar=False, c='b')
    
    
a["GrLivArea"].sort_values(ascending= False).head(2)
a["LotArea"].sort_values(ascending = False).head(2)
a["TotalBsmtSF"].sort_values(ascending = False).head(1)
a["1stFlrSF"].sort_values(ascending = False).head(1)

train=Train[(Train['GrLivArea'] < 4600) & (Train['MasVnrArea'] < 1500)]
print('We removed ',Train.shape[0]- train.shape[0],'outliers')
train.head()


target=a[['SalePrice']]
target.loc[691]
target.loc[1169]
pos=[1169,691,451]
target.drop(target.index[pos], inplace = True)


print('We make sure that both train and target sets have the same row number after removing the outliers:')
print( 'Train: ',train.shape[0], 'rows')
print('Target:', target.shape[0],'rows')

train.head()
target.head()


















































