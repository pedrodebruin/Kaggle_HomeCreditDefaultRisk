# Render our plots inline in IPython
#%matplotlib inline

import os

import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
plt.style.use('classic')
import re
import string
import random 

# File with preprocessing functions
import HCDR_preprocessing as pp

# Useful for the models dictionary
from collections import OrderedDict

# Some general options
debug = False # just a quick verbosity switch

def explore(Nrows = None):

        # Description and sample submission
        description = pd.read_csv('./data/HomeCredit_columns_description.csv')
        df_sample_sub = pd.read_csv('./data/sample_submission.csv')
        
	# Load the actual data
        df_full_test = pd.read_csv('./data/application_test.csv', nrows=Nrows)
        df_full_train = pd.read_csv('./data/application_train.csv', nrows=Nrows)
        df_bureau_bal = pd.read_csv('./data/bureau_balance.csv', nrows=Nrows)
        df_bureau = pd.read_csv('./data/bureau.csv', nrows=Nrows)
        df_cc_bal = pd.read_csv('./data/credit_card_balance.csv', nrows=Nrows)
        df_inst_pay = pd.read_csv('./data/installments_payments.csv', nrows=Nrows)
        df_POS_CASH_bal = pd.read_csv('./data/POS_CASH_balance.csv', nrows=Nrows)
        df_prev_app = pd.read_csv('./data/previous_application.csv', nrows=Nrows)

        # Merging of train data:
        extradatalist = [ df_bureau, df_bureau_bal, df_prev_app, df_cc_bal, df_inst_pay, df_POS_CASH_bal ]
        extradatakeylist = [ ['SK_ID_CURR'], ['SK_ID_BUREAU'], ['SK_ID_CURR'], ['SK_ID_PREV','SK_ID_CURR'], ['SK_ID_PREV','SK_ID_CURR'], ['SK_ID_PREV','SK_ID_CURR', 'SK_DPD', 'SK_DPD_DEF'] ]
        for i in range(0,6):
#        for i in range(0,2):
            extradf = extradatalist[i]
            extradfkeys = extradatakeylist[i]
            cols_to_use = extradf.columns.difference(df_full_train.columns).tolist()
            cols_to_use += extradfkeys
            print("i={}, cols_to_use: {}".format(i,cols_to_use))
            df2 = extradf.drop_duplicates(subset=extradfkeys)
            df2 = df2[cols_to_use]
            df_full_train = df_full_train.merge(df2, how='left', on=extradfkeys, validate='many_to_one')
            df_full_test = df_full_test.merge(df2, how='left', on=extradfkeys, validate='many_to_one')

        print("Shape of df_full_train: {0}".format(df_full_train.shape))
        print("Shape of df_full_test: {0}".format(df_full_test.shape))

	print("makeInputs: Head of train set:")
	print(df_full_train[:10])
 
        sister_cols = [col for col in df_full_train.columns if '_y' in col]
        print("These redundant columns were made after merging:\n{}".format(sister_cols))
	
	print("\nmakeInputs: Train data description:")
        ncols = df_full_train.shape[1]
        pp.blockprint(df_full_train)
        pp.printObjCols(df_full_train)
        pp.blockprint(df_full_train[ df_full_train.columns[df_full_train.dtypes=="object"] ], method="unique")

        allnan_cols = df_full_train.columns[ [np.all(df_full_train[c].isnull()) for c in df_full_train.columns] ] 
        print("\nColumns where all entries are NAN:\n{}".format(allnan_cols))
        print(df_full_train[allnan_cols].head())
	
        df_full_train.info()

	print("\nmakeInputs: Train data columns:")
	print(df_full_train.columns)
	print('')

#        print(df_full_train.corr()['TARGET'].sort_values(ascending=False).head(10))
#        print(df_full_train.corr()['TARGET'].sort_values(ascending=False).tail(10))

        print('\n')
        print(df_full_train['EXT_SOURCE_1'].dtypes)
        print(df_full_train['EXT_SOURCE_2'].dtypes)
        print(df_full_train['EXT_SOURCE_3'].dtypes)


#        ## IMPLEMENT SAFEGUARD TO MAKE SURE PREPROCESSING HASN'T SCRAMBLED ENTRIES
#        n_xcheck = 3
#        temp = df_full_train.loc[df_full_train['TARGET']==1,'SK_ID_CURR']
#        print("temp shape:{}".format(temp.shape))
#        print("temp head:\n{}".format(temp.head()))
#        random_entries = random.sample(df_full_train.loc[df_full_train['TARGET']==1,'SK_ID_CURR'].values, n_xcheck)
#        random_xcheck_before = {}
#        random_xcheck_after = {}
#        xcheck_col = 'TARGET' # other columns get scaled or converted to label...
#        for i in range(0,n_xcheck):
#            x = random_entries[i]
#            random_xcheck_before[x] = df_full_train.loc[df_full_train['SK_ID_CURR']==x,xcheck_col].values[0]
#            print("Sample entry {} with xcheck value {}".format(x, random_xcheck_before[x]) )
#        print('random_xcheck_before = {}'.format(random_xcheck_before))
#        ##### END OF CROSS-CHECK ON SCRAMBLING ####
	

def main():

    explore()
#    explore(Nrows = 100000)



if __name__=='__main__':
	main()


