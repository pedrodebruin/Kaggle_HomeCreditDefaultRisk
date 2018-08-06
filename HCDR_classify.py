# Render our plots inline in IPython
#%matplotlib inline

import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
plt.style.use('classic')
import re
import string
import glob

# Useful for the models dictionary
from collections import OrderedDict

# Classifiers and useful transforms
from sklearn import decomposition
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report,confusion_matrix,f1_score, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import lightgbm as lgb
import gc

# Some general options
debug = False # just a quick verbosity switch
useAllTrainingData = False

optimize_NN_activation = False
optimize_NN_alpha = False
optimize_NN_layer = False
optimize_SVC_C = False
optimize_SVC_kernel = False
optimize_RandFor = False
optimize_AdaBoost = False

# Threshold for predicting default
defaultThreshold = 0.5

def lgb_model(features, test_features, encoding = 'ohe', n_folds = 5):
    
    """Train and test a light gradient boosting model using
    cross validation. 
    
    Parameters
    --------
        features (pd.DataFrame): 
            dataframe of training features to use 
            for training a model. Must include the TARGET column.
        test_features (pd.DataFrame): 
            dataframe of testing features to use
            for making predictions with the model. 
        encoding (str, default = 'ohe'): 
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
            n_folds (int, default = 5): number of folds to use for cross validation
        
    Return
    --------
        submission (pd.DataFrame): 
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame): 
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame): 
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.
        
    """
    
    # Extract the ids
    print(features.columns)
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    
    # Extract the labels for training
    labels = features['TARGET']
    
    # Remove the ids and target
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    
    
    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        
        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join = 'inner', axis = 1)
        
        # No categorical indices to record
        cat_indices = 'auto'
    
    # Integer label encoding
    elif encoding == 'le':
        
        # Create a label encoder
        label_encoder = LabelEncoder()
        
        # List for storing categorical indices
        cat_indices = []
        
        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)
    
    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
        
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    
    # Extract feature names
    feature_names = list(features.columns)
    
    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)
    
    # Create the kfold object
    k_fold = KFold(n_splits = n_folds, shuffle = True)
    
    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    
    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])
    
    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []

    # What type of metric to use
    metric_str = 'binary_logloss'
    
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):
        
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]

#        lg = lgb.LGBMClassifier(n_estimators=10000, 
#                                   objective = 'binary', 
##                                   class_weight = 'balanced', 
#                                   reg_alpha = 0.1, reg_lambda = 0.1, 
#                                   subsample = 0.8, n_jobs = -1, verbose=200)
#
#        param_dist = {"max_depth": [-1],
#                      "learning_rate" : [0.5],
#                      "num_leaves": [30, 40, 50]
#        }
#        grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 5, scoring="roc_auc", verbose=5)
#        grid_search.fit(train_features, train_labels, 
#                 eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
#                 eval_names = ['valid', 'train'], categorical_feature = cat_indices)
#
#        # Record the best iteration
#        model = grid_search.best_estimator_
#        best_iteration = model.best_iteration_
        
        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   class_weight = 'balanced', 
                                   learning_rate = 1.,
                                   feature_fraction = 0.1,
                                   num_leaves = 100,
                                   max_depth = 15, 
                                   reg_alpha = 1., reg_lambda = 1., 
                                   subsample = 0.8, n_jobs = -1)
        
        # Train the model
        model.fit(train_features, train_labels, eval_metric = metric_str,
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 200)

        best_iteration = model.best_iteration_
        
        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        
        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
        
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        # Record the best score
        valid_score = model.best_score_['valid'][metric_str]
        train_score = model.best_score_['train'][metric_str]
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
        
    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)
    
    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances, metrics


def plot_scores( dflist, targetCol, name="" ):
    
    fig, ax = plt.subplots()
    colors = ['red', 'blue', 'green', 'black']
    for i,df in enumerate(dflist):
         ax.hist( df[targetCol+'_prob'], color=colors[i], density=True )

    ax.legend()
    
    plt.savefig('plots/'+name+'.png')
    

def classify(inputFolders, doPCA):

        df_full_train = pd.DataFrame()
        df_full_xval = pd.DataFrame()
        df_full_test = pd.DataFrame()
       
        NROWS = None
#        NROWS = 10000

        targetCol = 'TARGET'
        idxCol = 'SK_ID_CURR'

        dftypes = ['train', 'xval', 'test' ]
        for t in dftypes:
            list_ = []
            allFiles = []
            frame = pd.DataFrame()
            for d in inputFolders:
                allFiles.append( d+"/prepdf_full_"+t+".csv" )

            for file_ in allFiles:
                print ("Reading file {0} of type {1}".format(file_, t))
                df = pd.read_csv(file_, header=0, nrows = NROWS)
                list_.append(df)

            frame = pd.concat(list_, ignore_index=True)

            if t=='train':
                df_full_train = frame
            elif t=='xval':
                df_full_xval = frame
            elif t=='test':
                df_full_test = frame
        
            del frame 

        # Split full df into x and y dfs
        df_train_x = df_full_train
        df_train_x = df_train_x.drop(columns=[targetCol])
        s_train_y = df_full_train[targetCol]
        df_train_y = pd.DataFrame(df_full_train[targetCol], columns=[targetCol])
        df_xval_x  = df_full_xval
        df_xval_x  = df_xval_x.drop(columns=[targetCol])
        s_xval_y  = df_full_xval[targetCol]
        df_xval_y  = pd.DataFrame(df_full_xval[targetCol], columns=[targetCol])
        df_test_x  = df_full_test

        print( "\n\nclassify: Head of df_train_x:\n{0}".format(df_train_x.head()) )
        print( "\n\nclassify: Head of df_train_y:\n{0}".format(df_train_y.head()) )
        print( "\n\nclassify: Head of df_xval_x:\n{0}".format(df_xval_x.head()) )
        print( "\n\nclassify: Head of df_xval_y:\n{0}".format(df_xval_y.head()) )
        print( "\n\nclassify: Head of df_test_x:\n{0}".format(df_test_x.head()) )

        for c in df_full_train.columns["_y" in df_full_train.columns or "_x" in df_full_train.columns]:
            print ("makeInputs: Found sister column {0}".format(c))

	print("\n")
	print("#"*100)
	print("Let's try several different kinds of classifier")
	print("#"*100)
	print("\n")

        # Lot's of features, let's use PCA 
        if doPCA:
            pca = decomposition.PCA(0.95, svd_solver='full')

	keys = []
	scores = []
        models = OrderedDict()
        # Regressors don't have predict_proba, make sure the modelname has 'Regressor' so the later if statement protects
        models = {
#                  'Logistic Regression': LogisticRegression(solver='sag', class_weight='balanced'), 
#        	  'DecisionTreeRegressor max10': DecisionTreeRegressor( max_features='auto', max_depth=10),
#        	  'DecisionTreeRegressor max20': DecisionTreeRegressor( max_features='auto', max_depth=20),
#        	  'DecisionTreeRegressor max30': DecisionTreeRegressor( max_features='auto', max_depth=30),
#        	  'DecisionTreeRegressor max40': DecisionTreeRegressor( max_features='auto', max_depth=40),
#        	  'DecisionTreeRegressor max50': DecisionTreeRegressor( max_features='auto', max_depth=50),
#        	  'BDT (adaboost15)': AdaBoostClassifier(n_estimators=15, learning_rate=0.25),
#                  'SGDClassifier_alpha10m8': SGDClassifier(loss = 'modified_huber', alpha=0.00000001, max_iter=500, class_weight='balanced'),
#                  'SGDClassifier_alpha10m7': SGDClassifier(loss = 'modified_huber', alpha=0.0000001, max_iter=500, class_weight='balanced'),
#                  'SGDClassifier_alpha10m6': SGDClassifier(loss = 'modified_huber', alpha=0.000001, max_iter=500, class_weight='balanced'),
#                  'SGDClassifier_alpha10m5': SGDClassifier(loss = 'modified_huber', alpha=0.00001, max_iter=500, class_weight='balanced'),
#                  'SGDClassifier_alpha10m4': SGDClassifier(loss = 'modified_huber', alpha=0.0001, max_iter=500, class_weight='balanced'),
#                  'Random Forest_n50_md5': RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=-1), 
#                  'Random Forest_n50_md10': RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1), 
#                  'Random Forest_n50_md15': RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1), 
#                  'Random Forest_n50_md15_mf20': RandomForestClassifier(n_estimators=50, max_depth=15, max_features = 20, n_jobs=-1), 
#                  'Random Forest_n50_md15_mf30': RandomForestClassifier(n_estimators=50, max_depth=15, max_features = 30, n_jobs=-1), 
#                  'Random Forest_n50_md15_mf40': RandomForestClassifier(n_estimators=50, max_depth=15, max_features = 40, n_jobs=-1), 
#                  'Random Forest_n100_md15_mf40': RandomForestClassifier(n_estimators=100, max_depth=15, max_features = 40, n_jobs=-1), 
                  'Random Forest_n100_md10_mfNone': RandomForestClassifier(n_estimators=100, max_depth=10, max_features = None, n_jobs=-1), 
                  'Random Forest_n150_md10_mfNone': RandomForestClassifier(n_estimators=150, max_depth=10, max_features = None, n_jobs=-1), 
                  'Random Forest_n200_md10_mfNone': RandomForestClassifier(n_estimators=200, max_depth=10, max_features = None, n_jobs=-1), 
#                  'Random Forest_n50_md5': RandomForestClassifier(n_estimators=50, n_jobs=-1), 
#                  'Random_Forest_n200_md10': RandomForestClassifier(n_estimators=200, class_weight="balanced", max_depth=10 ), 
#                  'Random_Forest_n200_md15': RandomForestClassifier(n_estimators=200, class_weight="balanced", max_depth=15 ), 
#                  'Random_Forest_n200_md20': RandomForestClassifier(n_estimators=200, class_weight="balanced", max_depth=20 ), 
#                  'Random_Forest_n200_md30': RandomForestClassifier(n_estimators=200, class_weight="balanced", max_depth=30 ), 
#                  'Random_Forest_n200_md40': RandomForestClassifier(n_estimators=200, class_weight="balanced", max_depth=40 ), 
#                  'Random_Forest_Regressor_n50_md10': RandomForestRegressor(n_estimators=50, max_depth=10 ), 
#                  'Random_Forest_Regressor_n200_md20': RandomForestRegressor(n_estimators=200, max_depth=20 ), 
#                  'Random_Forest_Regressor_n250_md25': RandomForestRegressor(n_estimators=250, max_depth=25 ), 
#        	  'K-Nearest Neighbors_5': KNeighborsClassifier(n_neighbors=5, weights='distance'),
#        	  'K-Nearest Neighbors_10': KNeighborsClassifier(n_neighbors=10, weights='distance'),
#        	  'K-Nearest Neighbors_15': KNeighborsClassifier(n_neighbors=15, weights='distance'),
#        	  'K-Nearest Neighbors_20': KNeighborsClassifier(n_neighbors=20, weights='distance'),
#        	  'K-Nearest Neighbors_30': KNeighborsClassifier(n_neighbors=30, weights='distance'),
#        	  'K-Nearest Neighbors_50': KNeighborsClassifier(n_neighbors=50, weights='distance'),
#                  'SVC_rbf': SVC(kernel='rbf', gamma='auto', C=1.0, class_weight="balanced", probability=True), 
#                  'SVC (unbalanced, rbf)': SVC(kernel='rbf', gamma='auto', C=1.0), 
#                  'Linear SVC': SVC(kernel='rbf', C=1.0, class_weight="balanced", probability=True), 
#                  'Nu SVC': NuSVC(), 
#        	  'Neural Network_2_10_alpha10m2': MLPClassifier(activation='identity', solver='lbfgs', alpha=0.01, hidden_layer_sizes=(2,10)),
#        	  'Neural Network_2_10_alpha1m1': MLPClassifier(activation='identity', solver='lbfgs', alpha=0.1, hidden_layer_sizes=(2,10)),
#        	  'Neural Network_3_10_alpha1': MLPClassifier(activation='identity', solver='lbfgs', alpha=1., hidden_layer_sizes=(3,20)),
#        	  'Neural Network_3_30_alpha1': MLPClassifier(activation='identity', solver='lbfgs', alpha=1., hidden_layer_sizes=(3,30)),
#        	  'Neural Network_2_10_alpha10': MLPClassifier(activation='identity', solver='lbfgs', alpha=10., hidden_layer_sizes=(2,10)),
#        	  'Neural Network_2_25_alpha1': MLPClassifier(activation='identity', solver='lbfgs', alpha=1, hidden_layer_sizes=(2,25)),
#        	  'Neural Network_3_10_alpha1': MLPClassifier(activation='identity', solver='lbfgs', alpha=1, hidden_layer_sizes=(3,10)),
#        	  'Neural Network_3_20_alpha1': MLPClassifier(activation='identity', solver='lbfgs', alpha=1, hidden_layer_sizes=(3,20)),
                 }


        # Parameter to be optimized. For now only have implemented 1-dimensional hyperparameter optimization
        paramlist = []

        if optimize_AdaBoost:
            paramlist = range(20,60,10)
            for p in paramlist:
                modelname = 'BDT (adaboost{0})'.format(p)
                models[modelname] = AdaBoostClassifier(n_estimators=p, learning_rate=0.5)
            for p in paramlist:
                modelname = 'BDT (adaboost{0}, alpha025)'.format(p)
                models[modelname] = AdaBoostClassifier(n_estimators=p, learning_rate=0.25)


        ####################################################################################################
        # Optimizing RandomForest
        ####################################################################################################
        if optimize_RandFor:
            paramlist = range(1,15, 2)
            for p in paramlist:
                    modelname = 'RandomForestRegressor_{0}'.format(p)
                    models[modelname] = RandomForestRegressor(n_estimators=p) 
            for p in paramlist:
                    modelname = 'RandomForestRegressor_md5_{0}'.format(p)
                    models[modelname] = RandomForestRegressor(n_estimators=p, max_depth=5) 

        ####################################################################################################
        # Optimizing SVC
        ####################################################################################################
        if optimize_SVC_C:
            paramlist = [ 0.01*x for x in range(1,100) ]
            for p in paramlist:
                    modelname = 'SVCrbf_{0}'.format(p)
                    models[modelname] = SVC(kernel='rbf', gamma='auto', C=p)

        if optimize_SVC_kernel:
            paramlist = [ 'linear', 'poly', 'rbf', 'sigmoid' ]
            for p in paramlist:
                    modelname = 'SVC_{0}'.format(p)
                    models[modelname] = SVC(kernel=p, gamma='auto', C=1.0)
        ####################################################################################################
        # Optimizing Neural Network
        ####################################################################################################
        if optimize_NN_activation:
            paramlist = [ 'relu', 'identity', 'tanh', 'logistic' ]
            for p in paramlist:
                    modelname = 'Neural Network alpha_{0}'.format(p)
                    models[modelname] = MLPClassifier(activation=p, solver='lbfgs', alpha=15, hidden_layer_sizes=(2,6))

        if optimize_NN_alpha:
            paramlist = [ 1e-5*pow(1.1,x) for x in range (1,50) ]
            for p in paramlist:
                    modelname = 'Neural Network alpha_{0}'.format(str(m_alpha))
                    models[modelname] = MLPClassifier(activation='relu', solver='lbfgs', alpha=p, hidden_layer_sizes=(2,6))

        if optimize_NN_layer:
            layer1list = range(2,10)
            layer2list = range(2,10)
            for l1 in layer1list:
                for l2 in layer2list:
                    paramlist.append(10*l1 + l2)
                    modelname = 'Neural Network layers_{0}{1}'.format(str(l1),str(l2))
                    models[modelname] = MLPClassifier(activation='relu', solver='lbfgs', alpha=15, hidden_layer_sizes=(l1,l2))
        ####################################################################################################

        # For plotting results
        train_F1 = []
        xval_F1 = []
        modellist = []

        print( "\nUnique values in TARGET column (train):\n{0}".format(s_train_y.unique()) )
        print( "\nUnique values in TARGET column (xval):\n{0} ".format(s_xval_y.unique() ) )

        print("\nAbout to run classifier")


        features = df_train_x.columns.tolist()
        features.remove(idxCol)

        print( "\nNumber of features used in the fit training: {0}".format(len(features)) )
 
	for modelname,model in models.items():
		print("\nTrying out classifier {0}".format(modelname))

                if doPCA:
                    pipe = Pipeline(steps=[('pca', pca), (modelname, model)])
                else:
                    pipe = Pipeline(steps=[(modelname, model)])

		if useAllTrainingData:
			frames_x = [ df_train_x, df_xval_x ]
			frames_y = [ df_train_y, df_xval_y ]
			fulltrain_x_df = pd.concat(frames_x)
			fulltrain_y_df = pd.concat(frames_y)
                        if debug:
			    print("Are any entries NA? {0}".format(np.any(np.isnan(fulltrain_x_df))) )
			    print("Are any entries NA? {0}".format(np.any(np.isnan(fulltrain_y_df))) )
			    print("Are all entries finite? {0}".format(np.all(np.isfinite(fulltrain_x_df))))
			    print("Are all entries finite? {0}".format(np.all(np.isfinite(fulltrain_y_df))))
			clf = pipe.fit(fulltrain_x_df[features], fulltrain_y_df)		
		else:
                        if debug:
			    print("Are any entries not NA? {0}".format(np.any(np.isnan(df_train_x))))
			    print("Are any entries not NA? {0}".format(np.any(np.isnan(df_train_y))))
			    print("Are all entries finite? {0}".format(np.all(np.isfinite(df_train_x))))
			    print("Are all entries finite? {0}".format(np.all(np.isfinite(df_train_y))))
			clf = pipe.fit(df_train_x[features], s_train_y)

                if debug:
                     print ("Head of df_train_y:\n{0}".format(df_train_y.head()) )

                # predict_proba returns 2 columsn, the probability for 0 and for 1. Use [:,1] to select the probability for binary outcome 1
                if 'Regressor' in modelname:
                    train_y_pred = pd.DataFrame( pipe.predict(df_train_x[features]), columns = [targetCol+'_prob'] )
                else:
                    train_y_pred = pd.DataFrame( pipe.predict_proba(df_train_x[features])[:,1], columns = [targetCol+'_prob'] )
                train_y_pred[targetCol+'_pred'] = train_y_pred[targetCol+'_prob'].apply(lambda x: 0 if x < defaultThreshold else 1)
                train_y_pred[idxCol] = df_train_x[[idxCol]]
                train_y_pred[targetCol+'_orig'] = df_train_y[ targetCol ]
                train_y_pred = train_y_pred[ [ idxCol, targetCol+'_prob', targetCol+'_pred', targetCol+'_orig' ] ]
                
                if "Regressor" in modelname:
                    xval_y_pred = pd.DataFrame( pipe.predict(df_xval_x[features]), columns = [targetCol+'_prob'] )
                else:
                    xval_y_pred = pd.DataFrame(pipe.predict_proba(df_xval_x[features])[:,1], columns = [targetCol+'_prob'])
                xval_y_pred[targetCol+'_pred'] = xval_y_pred[targetCol+'_prob'].apply(lambda x: 0 if x < defaultThreshold else 1)
                xval_y_pred[idxCol] = df_xval_x[[idxCol]]
                xval_y_pred[targetCol+'_orig'] = df_xval_y[ targetCol ]
                xval_y_pred = xval_y_pred[ [ idxCol, targetCol+'_prob', targetCol+'_pred', targetCol+'_orig' ] ]

                modellist.append(modelname)

                train_F1.append(f1_score(df_train_y[targetCol], train_y_pred[targetCol+'_pred']))
		print("\n\nClassification report on training set:")
                print( classification_report(df_train_y[targetCol], train_y_pred[targetCol+'_pred']) )

                xval_F1.append(f1_score(df_xval_y[targetCol], xval_y_pred[targetCol+'_pred']))
		print("\nClassification report on cross-validation set:")
		print( classification_report(df_xval_y[targetCol], xval_y_pred[targetCol+'_pred']) )
                print("acc_score: {}".format(accuracy_score(xval_y_pred[targetCol+'_orig'], xval_y_pred[targetCol+'_pred'])) )
                print("AUROC: {}".format(roc_auc_score(xval_y_pred[targetCol+'_orig'], xval_y_pred[targetCol+'_prob'])) )
		print("\n")
	
                # Test set stuff:
                if "Regressor" in modelname:
                    test_y_pred = pd.DataFrame( pipe.predict(df_test_x[features]), columns = [targetCol] )
                else:
                    test_y_pred = pd.DataFrame(pipe.predict_proba(df_test_x[features])[:,1] , columns = [targetCol] )
                test_y_pred[idxCol] = df_test_x[[idxCol]]
                test_y_pred = test_y_pred[ [idxCol, targetCol] ]

		outstring = modelname.replace(' ', '')
		outstring = outstring.replace('-','')

		train_y_pred.to_csv("predictions/prediction_train_"+outstring+".csv", index=False)
		xval_y_pred.to_csv("predictions/prediction_xval_"+outstring+".csv", index=False)
		test_y_pred.to_csv("predictions/prediction_test_"+outstring+".csv", index=False)

                plot_scores( [train_y_pred, xval_y_pred], 'TARGET', name = modelname )


        print("\nPerformance of all-zero prediction:")
        print(accuracy_score(np.zeros(xval_y_pred.shape[0]),xval_y_pred[targetCol+'_orig']))


