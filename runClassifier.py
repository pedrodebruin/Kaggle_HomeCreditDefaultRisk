# Render our plots inline in IPython
#%matplotlib inline

import numpy as np
import pandas as pd
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
plt.style.use('classic')
import re
import string

# Useful for the models dictionary
from collections import OrderedDict

# Classifiers and useful transforms
from sklearn import decomposition
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from sklearn.neural_network import MLPClassifier


# Some general options
debug = False # just a quick verbosity switch
NROWS = 10000
useAllTrainingData = False

optimize_NN_activation = False
optimize_NN_alpha = False
optimize_NN_layer = False
optimize_SVC_C = False
optimize_SVC_kernel = False
optimize_RandFor_nest = False

usePreparedInputs = False


def classify(inFolder='preparedData'):


        df_full_train = pd.read_csv(inFolder+'/prepdf_full_train.csv')
        df_full_xval = pd.read_csv(inFolder+'/prepdf_full_xval.csv')
        df_full_test = pd.read_csv(inFolder+'/prepdf_full_test.csv')


        # Split full df into x and y dfs
        df_train_x = df_full_train
        df_train_x = df_train_x.drop(columns=['TARGET'])
        df_train_y = df_full_train['TARGET']
        df_xval_x  = df_full_xval
        df_xval_x  = df_xval_x.drop(columns=['TARGET'])
        df_xval_y  = df_full_xval['TARGET']
        df_test_x  = df_full_test

	print("\n")
	print("#"*100)
	print("Let's try several different kinds of classifier")
	print("#"*100)
	print("\n")

        # Lot's of features, let's use PCA 
        pca = decomposition.PCA(0.95, svd_solver='full')

	keys = []
	scores = []
        models = OrderedDict()
        models = {'Logistic Regression': LogisticRegression(), 
        	  'Decision Tree': DecisionTreeClassifier( class_weight="balanced"),
                  'Random Forest': RandomForestClassifier(n_estimators=40, class_weight="balanced" ), 
#        	  'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=1),
                  'SVC (rbf)': SVC(kernel='rbf', gamma='auto', C=1.0, class_weight="balanced"), 
                  'SVC (unbalanced, rbf)': SVC(kernel='rbf', gamma='auto', C=1.0), 
                  'Linear SVC': LinearSVC(class_weight="balanced"), 
#                  'Nu SVC': NuSVC(), 
        	  'Neural Network': MLPClassifier(activation='identity', solver='lbfgs', alpha=15, hidden_layer_sizes=(3,10), random_state=1)}


        # Parameter to be optimized. For now only have implemented 1-dimensional hyperparameter optimization
        paramlist = []

        ####################################################################################################
        # Optimizing RandomForest
        ####################################################################################################
        if optimize_RandFor_nest:
            paramlist = range(2,60)
            for p in paramlist:
                    modelname = 'RandFor_{0}'.format(p)
                    models[modelname] = RandomForestClassifier(n_estimators=p) 

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
                    models[modelname] = MLPClassifier(activation=p, solver='lbfgs', alpha=15, hidden_layer_sizes=(2,6), random_state=1)

        if optimize_NN_alpha:
            paramlist = [ 1e-5*pow(1.1,x) for x in range (1,50) ]
            for p in paramlist:
                    modelname = 'Neural Network alpha_{0}'.format(str(m_alpha))
                    models[modelname] = MLPClassifier(activation='relu', solver='lbfgs', alpha=p, hidden_layer_sizes=(2,6), random_state=1)

        if optimize_NN_layer:
            layer1list = range(2,10)
            layer2list = range(2,10)
            for l1 in layer1list:
                for l2 in layer2list:
                    paramlist.append(10*l1 + l2)
                    modelname = 'Neural Network layers_{0}{1}'.format(str(l1),str(l2))
                    models[modelname] = MLPClassifier(activation='relu', solver='lbfgs', alpha=15, hidden_layer_sizes=(l1,l2), random_state=1)
        ####################################################################################################

        # For plotting results
        train_F1 = []
        xval_F1 = []
        modellist = []

        print("\nAbout to run classifier")

	for modelname,model in models.items():
		print("\nTrying out classifier {0}".format(modelname))
                pipe = Pipeline(steps=[('pca', pca), (modelname, model)])

		if useAllTrainingData:
			frames_x = [ df_train_x, df_xval_x ]
			frames_y = [ df_train_y, df_xval_y ]
			fulltrain_x_df = pd.concat(frames_x)
			fulltrain_y_df = pd.concat(frames_y)

#                        print("Description of fulltrain_x_df:\n{0}".format(fulltrain_x_df.describe()))
#                        print("Description of fulltrain_y_df:\n{0}".format(fulltrain_y_df.describe()))
			print("Are all entries not NA? {0}".format(np.any(np.isnan(fulltrain_x_df))) )
			print("Are all entries not NA? {0}".format(np.any(np.isnan(fulltrain_y_df))) )
			print("Are all entries finite? {0}".format(np.all(np.isfinite(fulltrain_x_df))))
			print("Are all entries finite? {0}".format(np.all(np.isfinite(fulltrain_y_df))))
			clf = pipe.fit(fulltrain_x_df, fulltrain_y_df)		
		else:
#                        print("Description of df_train_x:\n{0}".format(df_train_x.describe()))
#                        print("Description of df_train_y:\n{0}".format(df_train_y.describe()))
			print("Are all entries not NA? {0}".format(np.any(np.isnan(df_train_x))))
			print("Are all entries not NA? {0}".format(np.any(np.isnan(df_train_y))))
			print("Are all entries finite? {0}".format(np.all(np.isfinite(df_train_x))))
			print("Are all entries finite? {0}".format(np.all(np.isfinite(df_train_y))))
			clf = pipe.fit(df_train_x, df_train_y)
		train_y_prob = pipe.predict(df_train_x)  
		xval_y_prob = pipe.predict(df_xval_x)
		test_y_prob = pipe.predict(df_test_x)

                # In this case we care about the probability, but here's the logical prediction anyway
		train_y_pred = np.where(train_y_prob > 0.5, 1, 0)	
		xval_y_pred = np.where(xval_y_prob > 0.5, 1, 0)	
		test_y_pred = np.where(test_y_prob > 0.5, 1, 0)	

                modellist.append(modelname)
                train_F1.append(f1_score(df_train_y, train_y_prob))
                xval_F1.append(f1_score(df_xval_y, xval_y_prob))

		print("Classification report on training set:")
		print(classification_report(df_train_y, train_y_prob))
		print("\nClassification report on cross-validation set:")
		print(classification_report(df_xval_y, xval_y_prob))
		print("\n\n")
		#print(confusion_matrix(train_y_df, train_y_pred))
		#print(confusion_matrix(xval_y_df, xval_y_pred))
	
		pred_df = pd.DataFrame( { 'SK_ID_CURR': df_full_test['SK_ID_CURR'], 'TARGET': test_y_prob })
		outstring = modelname.replace(' ', '')
		outstring = outstring.replace('-','')
		pred_df.to_csv("data/prediction_"+outstring+".csv", index=False)

        plt.plot(modellist,train_F1, 'bo--', linewidth=2)
        plt.plot(modellist,xval_F1, 'rx--', linewidth=2)
	plt.xlabel('Model Used')
        plt.xticks(modellist, modellist, rotation='vertical')
	plt.ylabel('F1 score')
        plt.ylim(0.5, 1.0)
        plt.tight_layout()
        plt.show()
 


