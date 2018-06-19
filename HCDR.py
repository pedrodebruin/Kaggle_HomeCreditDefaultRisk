# Render our plots inline in IPython
#%matplotlib inline

import os

import pandas as pd
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
plt.style.use('classic')
import re
import string

# File with preprocessing functions
import HCDRpreprocessing as pp
import runClassifier

# Useful for the models dictionary
from collections import OrderedDict

# Let's use seaborn this time
import seaborn as sns


# Some general options
debug = False # just a quick verbosity switch
useAllTrainingData = False

optimize_NN_activation = False
optimize_NN_alpha = False
optimize_NN_layer = False
optimize_SVC_C = False
optimize_SVC_kernel = False
optimize_RandFor_nest = False

prepareInputs = True

def makeInputs(outFolder='preparedData', Nrows = 100000, skipN=None):

        # Description and sample submission
        description = pd.read_csv('./data/HomeCredit_columns_description.csv')
        df_sample_sub = pd.read_csv('./data/sample_submission.csv')

        
	# Load the actual data
        df_app_test = pd.read_csv('./data/application_test.csv', nrows=Nrows, skiprows=skipN)
        df_app_train = pd.read_csv('./data/application_train.csv', nrows=Nrows, skiprows=skipN)
        df_bureau_bal = pd.read_csv('./data/bureau_balance.csv', nrows=Nrows, skiprows=skipN)
        df_bureau = pd.read_csv('./data/bureau.csv', nrows=Nrows, skiprows=skipN)
        df_cc_bal = pd.read_csv('./data/credit_card_balance.csv', nrows=Nrows, skiprows=skipN)
        df_inst_pay = pd.read_csv('./data/installments_payments.csv', nrows=Nrows, skiprows=skipN)
        df_POS_CASH_bal = pd.read_csv('./data/POS_CASH_balance.csv', nrows=Nrows, skiprows=skipN)
        df_prev_app = pd.read_csv('./data/previous_application.csv', nrows=Nrows, skiprows=skipN)


        # Merge the dataframes by their primary/foreign keys. Use a left join since the app data seems to have a lot of features already
        df_full_train = df_app_train.merge(df_bureau, how='left', on='SK_ID_CURR')
        df_full_train = df_full_train.drop_duplicates(subset=['SK_ID_CURR'])
        df_full_train = df_full_train.merge(df_bureau_bal, how='left', on='SK_ID_BUREAU')
        df_full_train = df_full_train.drop_duplicates(subset=['SK_ID_CURR'])
        df_full_train = df_full_train.merge(df_prev_app, how='left', on='SK_ID_CURR')
        df_full_train = df_full_train.drop_duplicates(subset=['SK_ID_CURR'])
        df_full_train = df_full_train.merge(df_cc_bal, how='left', on=['SK_ID_PREV','SK_ID_CURR'])
        df_full_train = df_full_train.drop_duplicates(subset=['SK_ID_CURR'])
        df_full_train = df_full_train.merge(df_inst_pay, how='left', on=['SK_ID_PREV','SK_ID_CURR'])
        df_full_train = df_full_train.drop_duplicates(subset=['SK_ID_CURR'])
        df_full_train = df_full_train.merge(df_POS_CASH_bal, how='left', on=['SK_ID_PREV','SK_ID_CURR'])
        df_full_train = df_full_train.drop_duplicates(subset=['SK_ID_CURR'])
        #print(df_full_train.shape)

        print(df_full_train.describe())

        df_full_test = df_app_test.merge(df_bureau, how='left', on='SK_ID_CURR')
        df_full_test = df_full_test.drop_duplicates(subset=['SK_ID_CURR'])
        df_full_test = df_full_test.merge(df_bureau_bal, how='left', on='SK_ID_BUREAU')
        df_full_test = df_full_test.drop_duplicates(subset=['SK_ID_CURR'])
        df_full_test = df_full_test.merge(df_prev_app, how='left', on='SK_ID_CURR')
        df_full_test = df_full_test.drop_duplicates(subset=['SK_ID_CURR'])
        df_full_test = df_full_test.merge(df_cc_bal, how='left', on=['SK_ID_PREV','SK_ID_CURR'])
        df_full_test = df_full_test.drop_duplicates(subset=['SK_ID_CURR'])
        df_full_test = df_full_test.merge(df_inst_pay, how='left', on=['SK_ID_PREV','SK_ID_CURR'])
        df_full_test = df_full_test.drop_duplicates(subset=['SK_ID_CURR'])
        df_full_test = df_full_test.merge(df_POS_CASH_bal, how='left', on=['SK_ID_PREV','SK_ID_CURR'])
        df_full_test = df_full_test.drop_duplicates(subset=['SK_ID_CURR'])
#        print("Shape of df_full_test: {0}".format(df_full_test.shape))
#        print(df_full_test.describe())

	print("Head of train set:")
	print(df_full_train[:10])
	
	print("\nTrain data description:")
	print(df_full_train.describe())
	
	print("\nTrain data columns:")
	print(df_full_train.columns)
	print('')
	
        # Set 10% of training data for cross-validation
        df_full_xval  = df_full_train[ (df_full_train.SK_ID_CURR%10 == 0) ]
        df_full_train = df_full_train[ (df_full_train.SK_ID_CURR%10 != 0) ]

        # Execute preprocessing functions in HCDRpreprocessing.py
        print("\nPre-processing train data")
        df_full_train = pp.preprocess(df_full_train)
	print("\nAfter preprocessing, # of entries missing in train data: \n{0}".format(df_full_train.isnull().sum()))
        print("\nPre-processing xval data")
        df_full_xval = pp.preprocess(df_full_xval)
	print("\nAfter preprocessing, # of entries missing in xval data: \n{0}".format(df_full_xval.isnull().sum()))
        print("\nPre-processing test data")
        df_full_test = pp.preprocess(df_full_test)
	print("\nAfter preprocessing, # of entries missing in test data: \n{0}" .format(df_full_test.isnull().sum()))

        df_full_train.to_csv(outFolder+'/prepdf_full_train.csv', index=False) 
        df_full_xval.to_csv(outFolder+'/prepdf_full_xval.csv', index=False) 
        df_full_test.to_csv(outFolder+'/prepdf_full_test.csv', index=False)
	


def main():

    nrows = 100000
    skiprange = [ None, range(1,nrows), range(1,nrows*2), range(1,nrows*3) ]

    if prepareInputs:
        for r in skiprange:

            if r == None:
                outfolder = 'preparedData_{0}_rows_1_to_{0}'.format(nrows) 
                if not os.path.exists(outfolder):
                    os.makedirs(outfolder)
                makeInputs(outfolder, nrows, r)
            else:
                outfolder = 'preparedData_{0}_rows_gt_{1}'.format(nrows, r.last())
                if not os.path.exists(outfolder):
                    os.makedirs(outfolder)
                makeInputs(outfolder, nrows, r)

        print("\n")
        print ("#"*100)
        print ("\nEND OF PRE-PROCESSING OF DATA\n")
        print ("#"*100)
        print("\n")

    for r in skiprange:
        if r == None:
            runClassifier.classify('preparedData_{0}_rows_1_to_{0}'.format(nrows) ) 
        else:
            runClassifier.classify('preparedData_{0}_rows_gt_{1}'.format(r.last()) ) 


if __name__=='__main__':
	main()


