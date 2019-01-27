import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings("ignore")

def readTrain(file):
    cols = ['idnum', 'age', 'workerclass', 'interestincome', 
            'traveltimetowork', 'vehicleoccupancy', 'meansoftransport', 
            'marital', 'schoolenrollment', 'educationalattain', 
            'sex', 'workarrivaltime', 'hoursworkperweek', 
            'ancestry', 'degreefield', 'industryworkedin', 
            'wages']
    df = pd.read_csv(file,header=None, names=cols, na_values='?')
    df=df.astype(float)
    df['workerclass'].replace(np.nan,0,inplace=True)
    df['vehicleoccupancy'].replace(np.nan,77,inplace=True)
    df['meansoftransport'].replace(np.nan,0,inplace=True)
    df['schoolenrollment'].replace(np.nan,0,inplace=True)
    df['educationalattain'].replace(np.nan,0,inplace=True)
    df['workarrivaltime'].replace(np.nan,0,inplace=True)
    df['degreefield'].replace(np.nan,0,inplace=True)
    df['industryworkedin'].replace(np.nan,0,inplace=True)
    df['interestincome'].replace(np.nan,0,inplace=True)
    df['traveltimetowork'].replace(np.nan,0,inplace=True)
    df['hoursworkperweek'].replace(np.nan,0,inplace=True)
    df['wages'].replace(np.nan,0,inplace=True)
    return df

def readTest(file):
    cols = ['idnum', 'age', 'workerclass', 'interestincome', 
            'traveltimetowork', 'vehicleoccupancy', 'meansoftransport', 
            'marital', 'schoolenrollment', 'educationalattain', 
            'sex', 'workarrivaltime', 'hoursworkperweek', 
            'ancestry', 'degreefield', 'industryworkedin', 
            'wages']
    df = pd.read_csv(file,header=None, names=cols, na_values='?')
    df=df.astype(float)
    df['workerclass'].replace(np.nan,0,inplace=True)
    df['vehicleoccupancy'].replace(np.nan,77,inplace=True)
    df['meansoftransport'].replace(np.nan,0,inplace=True)
    df['schoolenrollment'].replace(np.nan,0,inplace=True)
    df['educationalattain'].replace(np.nan,0,inplace=True)
    df['workarrivaltime'].replace(np.nan,0,inplace=True)
    df['degreefield'].replace(np.nan,0,inplace=True)
    df['industryworkedin'].replace(np.nan,0,inplace=True)
    df['interestincome'].replace(np.nan,0,inplace=True)
    df['traveltimetowork'].replace(np.nan,0,inplace=True)
    df['hoursworkperweek'].replace(np.nan,0,inplace=True)
    
    return df

def onehot(df):
    
    bins_i = [-1, 169,290,490, 690, 770,3990,4590,5790,6390,6780,7190,7790,7890,8290,8470,8690,9290,9590,9870,9920]
    names_i = ['No Industry', 'AGR', 'EXT', 'UTL', 'CON','MFG','WHL','RET','TRN','INF','FIN','PRF','EDU',
             'MED','SCA','ENT','SRV','EDM','MIL','never_Worked']
    df['industry']=pd.cut(df['industryworkedin'],bins_i,labels=names_i)
    
    bins_w = [-2,0,70,142,214,285]
    names_w=['nowork','late_night','morning','afternoon','night']
    df['workarrival']=pd.cut(df['workarrivaltime'],bins_w,labels=names_w)
    
    workerclass = ['workerclass_0.0','workerclass_1.0','workerclass_2.0','workerclass_3.0','workerclass_4.0',
               'workerclass_5.0','workerclass_6.0','workerclass_7.0','workerclass_8.0','workerclass_9.0']
    meansoftransport = ['meansoftransport_0.0',
                    'meansoftransport_1.0',
                    'meansoftransport_2.0',
                    'meansoftransport_3.0',
                    'meansoftransport_4.0',
                    'meansoftransport_5.0',
                    'meansoftransport_6.0',
                    'meansoftransport_7.0',
                    'meansoftransport_8.0',
                    'meansoftransport_9.0',
                    'meansoftransport_10.0',
                    'meansoftransport_11.0',
                    'meansoftransport_12.0'
                   ]

    schoolenrollment = [
                    'schoolenrollment_0.0',
                    'schoolenrollment_1.0',
                    'schoolenrollment_2.0',
                    'schoolenrollment_3.0'
                    ]
    educationalattain = [
                    'educationalattain_0.0',
                    'educationalattain_15.0',
                    'educationalattain_16.0',
                    'educationalattain_17.0',
                    'educationalattain_18.0',
                    'educationalattain_19.0',
                    'educationalattain_20.0',
                    'educationalattain_21.0',
                    'educationalattain_22.0',
                    'educationalattain_23.0',
                    'educationalattain_24.0'
                    ]
    marital = [
            'marital_1.0',
            'marital_2.0',
            'marital_3.0',
            'marital_4.0',
            'marital_5.0'
            ]
    
    industry = ['industry_NoIndustry', 'industry_AGR', 'industry_EXT', 'industry_UTL', 'industry_CON','industry_MFG',
            'industry_WHL','industry_RET','industry_TRN','industry_INF','industry_FIN','industry_PRF','industry_EDU',
         'industry_MED','industry_SCA','industry_ENT','industry_SRV','industry_EDM','industry_MIL','industry_never_Worked']
    
    workarrival = ['workarrival_nowork','workarrival_late_night','workarrival_morning','workarrival_aftrenoon',
                   'workarrival_night_shift']    
    
    cat_cols = [workerclass,meansoftransport,schoolenrollment,educationalattain,marital,workarrival,industry]
    categorical = ['workerclass','meansoftransport','schoolenrollment','educationalattain','marital','workarrival',
                   'industry']
    dicts={}
    c1=0
    for c in categorical:
        dicts['df_'+c] = pd.concat((pd.get_dummies(df[c],prefix=c), pd.DataFrame(columns=cat_cols[c1]))).fillna(0)
        c1+=1
   
    df_workerclass = pd.DataFrame.from_dict(dicts['df_workerclass'])
    df_meansoftransport = pd.DataFrame.from_dict(dicts['df_meansoftransport'])
    df_schoolenrollment = pd.DataFrame.from_dict(dicts['df_schoolenrollment'])
    df_educationalattain = pd.DataFrame.from_dict(dicts['df_educationalattain'])
    df_marital = pd.DataFrame.from_dict(dicts['df_marital'])
    df_workarrivaltime = pd.DataFrame.from_dict(dicts['df_workarrival'])
    df_industryworkedin = pd.DataFrame.from_dict(dicts['df_industry'])
    frames = [df_workerclass,df_meansoftransport,df_schoolenrollment,df_educationalattain,df_marital,
                df_workarrivaltime,df_industryworkedin]
    df_categorical = pd.concat(frames,axis=1)
    return df_categorical

def numerical(df):
    df_numerical = df[['sex','idnum','age','interestincome','traveltimetowork','hoursworkperweek','wages']]
    df_numerical.set_index('idnum')
    data = df_numerical[['age','traveltimetowork','hoursworkperweek']]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_data_df = pd.DataFrame(scaled_data, index=data.index, columns=data.columns)
    scaled_data_df.head()
    return scaled_data_df

def lasso(X_train,y_train,X_test,y_test,alphas):
    lasso = Lasso(max_iter=10000)
    lassocv = LassoCV(alphas=alphas, cv=10, max_iter=10000)
    lassocv.fit(X_train, y_train)
    lasso.set_params(alpha=lassocv.alpha_)
    alpha=lassocv.alpha_
    lasso.fit(X_train, y_train)
    mse = mean_squared_error(y_test, lasso.predict(X_test))
    rmse = np.sqrt(mean_squared_error(y_test, lasso.predict(X_test)))
    coef = pd.Series(lasso.coef_, index=X_test.columns)
    return alpha,mse,rmse,coef

def ridge(X_train,y_train,X_test,y_test,alphas):
    ridgecv = RidgeCV(alphas=alphas,normalize=False,cv=10)
    ridgecv.fit(X_train, y_train)
    alpha=ridgecv.alpha_
    ridge6 = Ridge(alpha=ridgecv.alpha_,normalize=False)
    ridge6.fit(X_train, y_train)
    mse = mean_squared_error(y_test, ridge6.predict(X_test))
    rmse = np.sqrt(mean_squared_error(y_test, ridge6.predict(X_test)))
    coef = pd.Series(ridge6.coef_[0], index=X_test.columns)
    return alpha,mse,rmse,coef

def grid():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    return random_grid


def main():
    df_train = readTrain('census_train.csv')
    df_test = readTest('census_test.csv')
    
    train_categorical = onehot(df_train)
    train_numerical = numerical(df_train)
    X_train = pd.concat((train_numerical,df_train['sex'],train_categorical),axis=1)
    y_train = df_train['wages']
    
    test_categorical = onehot(df_test)
    test_numerical = numerical(df_test)
    X_test = pd.concat((test_numerical,df_test['sex'],test_categorical),axis=1)
    
#     X_train,X_test,y_train,y_test=train_test_split(X_train,df_train['wages'],test_size=0.3,random_state=3)
    alphas = 10**np.linspace(6,-2,50)*0.5
#     l_alpha,l_mse,lrmse,l_coef = lasso(X_train,y_train,X_test,y_test,alphas)
#     r_alpha,r_mse,rmse,r_coef = ridge(X_train,y_train,X_test,y_test,alphas)

    rf = RandomForestRegressor()
    random_grid = grid()
    print('Finding hyperparameter for random forest regressor ...')
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                               n_iter = 100, cv = 3,verbose=1, random_state=42, n_jobs = -1)

    rf_random.fit(X_train, y_train)
    best_random = rf_random.best_estimator_
    print('Generating predictions...')
    preds = best_random.predict(X_test)
    results = pd.DataFrame({
    "Id" : df_test['idnum'],
    "Wages" : preds,
    })
    results.to_csv('test_outputs.csv',index=False)
    print('Predictions saved in test_outputs.csv in the program folder.')

if __name__ == '__main__':
    main()