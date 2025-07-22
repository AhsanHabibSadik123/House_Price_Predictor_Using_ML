import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression ,Ridge,Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('Bengaluru_House_Data.csv')
df1 = df.drop(['availability', 'area_type', 'society', 'balcony'], axis='columns')
df2 = df1.dropna().copy()
df2.loc[:, 'bedrooms'] = df2['size'].apply(lambda x: x.split(' ')[0])
df2.loc[:, 'bedrooms'] = df2['bedrooms'].astype(int)
df3 = df2.drop(['size'], axis='columns')
def range_to_sqrt(x):
    token = x.split('-')
    if len(token) == 2:
        return (float(token[0]) + float(token[1])) / 2
    try:
        return float(x)
    except:
        return None
df4 = df3.copy()
df4.loc[:, 'total_sqft'] = df3['total_sqft'].apply(range_to_sqrt)
df5 = df4.copy()
df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']

df5.location = df5.location.apply(lambda x : x.strip())
location_stat = df5.location.value_counts()

location_stat_less_than_10 = location_stat[location_stat <= 10]
df5.location = df5.location.apply(lambda x : 'other' if x in location_stat_less_than_10 else x)
very_small_room = df5[df5.total_sqft / df5.bedrooms < 300]

df6 = df5[~(df5.total_sqft / df5.bedrooms < 300)]

def removing_outliers_pps(df):
    df_out = pd.DataFrame()
    for key , sub_df in df.groupby('location'):
        mean = np.mean(sub_df.price_per_sqft)
        std = np.std(sub_df.price_per_sqft)
        reduced_df = sub_df[(sub_df.price_per_sqft > (mean - std)) & (sub_df.price_per_sqft < (mean + std))]
        df_out = pd.concat([reduced_df , df_out] , ignore_index=True)
    return df_out
df7 = removing_outliers_pps(df6)

def remove_bhk_outlier(df):
    outlier = []
    for location , location_df in df.groupby('location'):
        bhk_price_stats = {}
        for bhk , bhk_df in location_df.groupby('bedrooms'):
            bhk_price_stats[bhk] = {
                'mean' : np.mean(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk , bhk_df in location_df.groupby('bedrooms'):
            if bhk - 1 in bhk_price_stats and bhk_price_stats[bhk - 1]['count'] > 5:
                lower_bhk_mean = bhk_price_stats[bhk - 1]['mean']
                outlier.extend(bhk_df[bhk_df['price_per_sqft'] < lower_bhk_mean].index)
    return df.drop(outlier)

df8 = remove_bhk_outlier(df7)
df9 = df8[~(df8.bath > df8.bedrooms + 2)]
df10 = df9.drop('price_per_sqft',axis='columns')
dummies = pd.get_dummies(df10.location )
df11 = pd.concat([df10.drop('location' , axis='columns') , dummies.drop('other' , axis='columns')] , axis='columns')
X = df11.drop('price', axis='columns')
y = df11.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.tree import DecisionTreeRegressor
# model_params = {
#     "Linear Regression": {
#         "model": LinearRegression(),
#         "params": {
#             'fit_intercept': [True, False],
#             'positive': [True, False]
#         }
#     },
#     "Ridge Regression": {
#         "model": Ridge(),
#         "params": {
#             "alpha": [0.1, 1.0, 10.0, 100.0]
#         }
#     },
#     "Lasso Regression": {
#         "model": Lasso(),
#         "params": {
#             "alpha": [0.01, 0.1, 1.0, 10.0]
#         }
#     },
#     "Decision Tree": {
#         "model": DecisionTreeRegressor(),
#         "params": {
#             "max_depth": [None, 5, 10, 20],
#             "min_samples_split": [2, 5, 10]
#         }
#     }
# }
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# models = []
# for key, mp in model_params.items():
#     print(f"\nRunning GridSearchCV for: {key}")
#     model = GridSearchCV(mp['model'], mp['params'], cv=10)
#     model.fit(X_train, y_train)
#     best_estimator = model.best_estimator_
#     score = best_estimator.score(X_test, y_test)
#     print(f"Best Parameters: {model.best_params_}")
#     print(f"Test Accuracy: {score:.4f}")

model = LinearRegression(fit_intercept=False , positive=False)
model.fit(X , y)

def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    
    loc_index_array = np.where(X.columns == location)[0]
    if len(loc_index_array) > 0:
        x[loc_index_array[0]] = 1
    
    return model.predict([x])[0]

print(predict_price("1st Phase JP Nagar" , 1000 , 3 , 3))    

import joblib
joblib.dump(model , 'bangalore_home_price_model.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')