#This file contains machine learning tools that perform the classification and evaluation tasks.

import sys
import data_handler as dh
import numpy as np
import visualizer as vis
from sklearn import linear_model
from sklearn import neural_network
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold 
from sklearn import preprocessing
from padasip import filters
from statistics import mean
import os

if not os.path.exists('./figures/'):
		os.makedirs('./figures/')
	

#Parsing the R3 data from CSV files
R3dataset = dh.r3parser()
#Creating a list of companies
companies = ['B365','BW','IW','LB']

#Running mean squared loss analysis using the FilterLMS class of padasip.
print("Mean Squared Loss Results:")
#Setting the points in the result space for each category.
winEnumeration = {'Home':1,'Draw':0,'Away':-1}
for companyIndex,company in enumerate(companies):
    kf = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = []
    analytics = {'Away':[],'Draw':[],'Home':[]}
    lengths = {'Away':0,'Draw':0,'Home':0}
    i=0
    for train_index, test_index in kf.split(R3dataset[companyIndex]):
        #splitting the dataset for the 10-fold analysis.
        train_data, test_data = R3dataset[companyIndex].loc[train_index], R3dataset[companyIndex].loc[test_index]
        reg = filters.FilterLMS(n=3,mu=0.007)
        results = []
        analytics_partial = {'Away':[],'Draw':[],'Home':[]}
        log_d = []
        log_y = []
        for index, row in train_data.iterrows():
            reg.adapt(winEnumeration[row['win']],np.array([float(row['h_chance']),float(row['d_chance']),float(row['a_chance'])]))
            y = reg.predict(np.array([float(row['h_chance']),float(row['d_chance']),float(row['a_chance'])]))
            if y > 0:
                log_y.append(min(3,y))
            else:
                log_y.append(max(-3,y))
            log_d.append(winEnumeration[row['win']])
        vis.showPlot(log_d,log_y,company+'_fold_'+str(i))
        i+=1
        for index, row in test_data.iterrows():
            res = reg.predict(np.array([float(row['h_chance']),float(row['d_chance']),float(row['a_chance'])]))
            #Setting the label according to the minimum distance from the three category points.
            dis_A = abs(winEnumeration['Away']-res)
            dis_H = abs(winEnumeration['Home']-res)
            dis_D = abs(winEnumeration['Draw']-res)
            if(dis_A < dis_H and dis_A < dis_D):
                res = 'Away'
            elif(dis_H < dis_A and dis_H < dis_D):
                res = 'Home'
            elif(dis_D < dis_A and dis_D < dis_H):
                res = 'Draw'
            #Marking the successes and failures of the predictor.
            if res == row['win']:
                results.append(1)
                analytics_partial[row['win']].append(1)
            else:
                results.append(0)
                analytics_partial[row['win']].append(0)              
        scores.append(sum(results)/len(results))
        for key in analytics:
            analytics[key].append(sum(analytics_partial[key])/len(analytics_partial[key]))
            lengths[key]+=len(analytics_partial[key])
    print('-----------------------------------------------------------------------------------')
    print('Analysis for '+company+'.')
    print()
    R3dataset[companyIndex][['h_chance','d_chance','a_chance']].info()
    print()
    print('%0.2f accuracy with a standard deviation of %0.2f' % (mean(scores), np.std(scores)))
    print('having %0.2f max accuracy and %0.2f min accuracy.' % (max(scores), min(scores)))
    print()
    
    for key in analytics:
        if len(analytics[key]) > 0:
            print('stats for '+key+ ' from '+str(lengths[key])+' samples.')
            print()
            print('%0.2f accuracy with a standard deviation of %0.2f' % (mean(analytics[key]), np.std(analytics[key])))
            print('having %0.2f max accuracy and %0.2f min accuracy.' % (max(analytics[key]), min(analytics[key])))
            print()
        else:
            print('stats for '+key)
            print()
            print('No samples available.')
            print()
    

#Running squared loss analysis using the SGDClassifier class of Scikit.
print("Squared Loss Results:")
for companyIndex,company in enumerate(companies):
    reg = linear_model.SGDClassifier(loss="squared_loss",random_state=1)
    scores = cross_val_score(reg, R3dataset[companyIndex][['h_chance','d_chance','a_chance']], R3dataset[companyIndex]['win'], cv=10)
    print('-----------------------------------------------------------------------------------')
    print('Analysis for '+company+'.')
    print()
    R3dataset[companyIndex][['h_chance','d_chance','a_chance']].info()
    print()
    print('%0.2f accuracy with a standard deviation of %0.2f' % (scores.mean(), scores.std()))
    print('having %0.2f max accuracy and %0.2f min accuracy.' % (max(scores), min(scores)))
    print()

#Parsing the R3 data from CSV files
R28dataset = dh.r28parser()

#Running DNN analysis using the MLPClassifier class of Scikit.
print()
print("Deep Neural Network Results:")
print()
R28dataset.loc[:, R28dataset.columns != 'win'].info()
print()
print(R28dataset.loc[:, R28dataset.columns != 'win'].head)
print()
print(R28dataset['win'])
print()

X = R28dataset.loc[:, R28dataset.columns != 'win']
y = R28dataset['win']
laEn = preprocessing.LabelEncoder()
laEn.fit(y)
#metatrepei se arithmous
yEncoded = laEn.transform(y)
best = neural_network.MLPClassifier(activation='identity', hidden_layer_sizes=7, learning_rate='constant', max_iter=2000, random_state=1, solver='adam')
#Run hyperparameter optimization to pick the best DNN model.
if len(sys.argv) > 1 and sys.argv[1] == "hyperparameters":
    parameter_space = {
        'hidden_layer_sizes': [
            (int(X.shape[1]/2),int(X.shape[1]/4),int(X.shape[1]/8),int(X.shape[1]/16)),
            (int(X.shape[1]/2),int(X.shape[1]/4),int(X.shape[1]/8)),
            (int(X.shape[1]/4),int(X.shape[1]/8),int(X.shape[1]/16)),
            (int(X.shape[1]/2),int(X.shape[1]/8),int(X.shape[1]/16)),
            (int(X.shape[1]/2),int(X.shape[1]/4)),
            (int(X.shape[1]/4),int(X.shape[1]/8)),
            (int(X.shape[1]/8),int(X.shape[1]/16)),
            (int(X.shape[1]/2)),
            (int(X.shape[1]/4)),
            (int(X.shape[1]/8))
        ],
        'random_state':[1],
        'max_iter':[2000],
        'activation': ['identity','logistic','tanh','relu'],
        'solver': ['sgd','adam'],
        'learning_rate': ['constant','adaptive']
    }
#parametrous
    clf = GridSearchCV(best, parameter_space, n_jobs=-1, cv=10,verbose=1,scoring="r2")
    best = clf.fit(X, yEncoded)
    print('Best Model: %0.3f (+/-%0.03f) for %r\n' % (clf.cv_results_['mean_test_score'][clf.best_index_],clf.cv_results_['std_test_score'][clf.best_index_]*2,clf.cv_results_['params'][clf.best_index_]))

#Test again the best (or default) model and print statistics.
scores = cross_val_score(best, X, yEncoded, cv=10)
print()
print('%0.2f accuracy with a standard deviation of %0.2f' % (scores.mean(), scores.std()))
print('having %0.2f max accuracy and %0.2f min accuracy.' % (max(scores), min(scores)))
print()
