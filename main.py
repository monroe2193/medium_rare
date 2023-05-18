from IPython.display import display
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as p
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
from plotly.offline import plot, iplot, init_notebook_mode
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv("/Users/Jorden/PycharmProjects/Insuranceproject/insurance.csv")
print(f"There a {data.shape[0]} observations and {data.shape[1]} columns in the data set")
print(f"There are {data.isna().sum().sum()} missing values from the data")

#Numerical Data-types
pd.set_option('display.max_columns', None)
print(data.describe().T)

#Object Data-types
#Smoker data is yes/no format, want to show: Smoker/Non-Smoker
data['Smoker'] = data['Smoker'].apply(lambda x: 'Smoker' if x == 'yes' else 'Non-Smoker')
print(data.select_dtypes(include='object').describe().T)

#Of the object Data-types, find the avg charge(insurance payout) for each of Sex, Smoker/Non, Region
catagories_lst = data.select_dtypes(include=['object']).columns.tolist()
print(catagories_lst)
for column in catagories_lst:
    observation = data[column].value_counts()
    avg_claim = data.groupby(column)['Charges'].mean()
    title1 = "Number of Policy Holders"
    title2 = "Average Claim Amount"
    display(pd.DataFrame({title1: observation, title2: avg_claim})\
    .sort_values(title1, ascending=False))

#Insurance Claim Amount Based on Region
#sns.boxplot(x='Region', y='Charges', data=data)\
    #.set(title="Insurance Claim Amount Based on Region")

#Insurance Claim Amount Based on Smoking Status
#sns.boxplot(x='Smoker', y='Charges', data=data)\
    #.set(title="Insurance Claim Amount Based on Smoking Status")

#Probabilty of Different Claim Size Depending on Smoking Status
#sns.displot(data, x='Charges', hue='Smoker', stat='probability')\
    #.set(title='Probabilty of Different Claim Size Depending on Smoking Status')

#Split into age groups, find avg claim amount, plot based on smoking-status/age
#group_df = data.copy()
#group_df["Age_Group"]=['18 to 29 years' if i<30 else '30 to 44 years' if (i>=30)&(i<45) else
#                      '45 to 59 years' if (i>=45)&(i<60) else '60 and over' for i in data['Age']]
#group_df = group_df.groupby(['Age_Group','Smoker'])['Charges'].mean()
#group_df = group_df.rename('Charges').reset_index().sort_values('Smoker', ascending=True)
#sns.barplot(group_df, x='Age_Group', y='Charges', hue='Smoker')\
    #.set(title='Average Claim Cost by Age Group and Smoking Status')

#sns.lineplot(data, x='Age', y='Charges', hue='Smoker')\
    #.set(title='Claim Cost by Age Group and Smoking Status')

sns.scatterplot(data, x='BMI', y='Charges', hue='Smoker')\
    .set(title='Claim Cost relation to BMI Seperated by SMoking Status')

#data['Region'] = data['Region'].apply(lambda x: 'NE' if x == 'northeast' else \
    #('NW' if x=='northwest' else ('SE' if x=='southeast' else 'SW')))

#group_plot = data.groupby(['Region', 'Sex', 'Smoker'])['Charges'].mean()
#print(group_plot)

#group_plot = group_plot.rename('Charges').reset_index()

#g = sns.FacetGrid(group_plot, col='Region', row='Smoker', hue='Smoker',legend_out=True)
    #.set_titles(row_template=)
#g.map_dataframe(sns.barplot, y='Charges')


#corrmatrix = data.corr()
#print(corrmatrix)

p.show()
