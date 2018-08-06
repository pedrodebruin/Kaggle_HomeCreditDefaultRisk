
# This will be useful to convert letter to ints
from string import ascii_lowercase, ascii_uppercase
import os
import json

# For mean/dev scaling. Important for PCA, and good practice in general
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from collections import OrderedDict

import pandas as pd
import numpy as np

from collections import defaultdict

debug = False


def printObjCols(df, nrows=10):
    cols = df.columns
    objcols = df.columns[ df.dtypes=="object"]
    print("\nThese are the object columns in the dataframe:\n{}".format(df[objcols].head()))


def blockprint(df, method="head", ncols_block = 6, nrows=10):
    ncols = df.shape[1]
    i = 0
    while i<ncols:
        j = i + ncols_block
        if j >= ncols:
            j = ncols-1

        if method=="head":
            print(df[ df.columns[i:j] ].head(nrows) )
        elif method=="describe":
            print(df[ df.columns[i:j] ].describe() )
        elif method=="unique":
            for c in df.columns[i:j]:
                print("\nUnique values for column {}:".format(c))
                print(df[c].unique())
        

        i = j+1


def Scale(df, y='TARGET'):

    idxcol = []
    # Don't scale target column or index columns:
    for c in df.filter(regex='SK_ID*').columns:
        idxcol.append(c)
    if y in df.columns:
        idxcol.append(y)
    targetdf = pd.DataFrame(df[idxcol], columns=idxcol)

    df = df.drop(columns=idxcol)
    xcol = df.columns
    cols_in_scale = df.columns.tolist()

    # implemented model persistency here
    scaler = StandardScaler()
    if not os.path.exists('./persistent_models/scaler.pkl'):
        scaleddf = scaler.fit_transform(df)
        joblib.dump(scaler, 'scaler.pkl')
    else:
        scaler = joblib.load('./persistent_models/scaler.pkl')
        scaleddf = scaler.transform(df)
                
    scaleddf = pd.DataFrame(scaleddf, columns = xcol)
    scaleddf = pd.concat( [targetdf,scaleddf], axis=1 )

    return scaleddf, scaler, cols_in_scale


def makeNameTokens(list_df, col, name_dict):

        numbervalue = name_dict.values()[-1] + 1
        for df in list_df:
	        for i,row in df.iterrows():
	                name_cidx = df.columns.get_loc(col)
	                name_str = str(row[name_cidx])
	
	                if name_str not in name_dict.keys():
	                        if debug:
	                                print("Adding pair {0}:{1} to name dictionary".format(name_str, numbervalue))
	                        name_dict[name_str] = numbervalue
	                        numbervalue += 1

        return name_dict


def tokenizeNames(df,col,namedict):

    newcol = df[col].replace(namedict, regex=True)
    return newcol


def featureScaler(df_train, df_test, stringcols, numericcols, targetCol):
        df_train_string = pd.DataFrame(df_train[stringcols], columns=stringcols)
        df_train_num    = pd.DataFrame(df_train[numericcols], columns=numericcols)
        df_train_num, scalefit, cols_in_scalefit = Scale(df_train[numericcols])
        df_train = pd.concat([df_train_num,df_train_string], axis=1)

        # Now use the scale fit above for the test data (remember test data doesn't have targetCol):
        if targetCol in numericcols: numericcols.remove(targetCol)
        # there may be other columns that are present that the fit doesn't use:
        cols_dont_scale = [x for x in numericcols if x not in cols_in_scalefit]
        df_test_string = pd.DataFrame(df_test[stringcols], columns=stringcols)
        df_test_num    = pd.DataFrame(df_test[numericcols], columns=numericcols)
        df_test_temp = pd.DataFrame(df_test[cols_dont_scale], columns=cols_dont_scale)
        print("Going to remove {} from {}".format(cols_dont_scale,numericcols))
        for c in cols_dont_scale:
            numericcols.remove(c)
        if df_test.shape[0] > 0:
            df_test_num[numericcols] = scalefit.transform(df_test[numericcols])
        df_test = pd.concat([df_test_num,df_test_string], axis=1)

        return df_train,df_test


def encode_dfs(df_train, df_test, stringcols, numericcols, myEncoding=False, rebuildEncoders=False):

    # RECOMMENDED (uses pandas' built-in methods
    if not myEncoding:
        # OR simply label string columns
        clfs = {c:LabelEncoder() for c in stringcols}
        c_ohe = {c:OneHotEncoder() for c in stringcols}

        le_count = 0 # count columns encoded with LabelEncoder
        for col, clf in clfs.items():

            # Don't encode index columns -- shouldn't happen as they are not numerical, but not a bad guard
            if "SK_" in col: continue

            combined_col = pd.concat([df_train[col],df_test[col]])
            encoderpath = './encoders/'+col+'.pkl'

            # For categorical variables with less than 3 unique values, do a simple labeling
            if len(list(df_train[col].unique())) <= 2: 
                if not os.path.exists(encoderpath):
                    print("Encoding column {} with file {}".format(col, encoderpath))
                    clf.fit(combined_col)
                    df_train[col] = clf.transform(df_train[col])
                    df_test[col] = clf.transform(df_test[col])
                    joblib.dump(clf, encoderpath)
                else:
                    print("Encoding column {} with file {}".format(col, encoderpath))
                    clf = joblib.load(encoderpath)
                    df_train[col] = clf.transform(df_train[col])
                    df_test[col] = clf.transform(df_test[col])
           
                le_count += 1

    print("Converted {} columns using LabelEncoder()".format(le_count))

    df_train = pd.get_dummies(df_train)
    df_test  = pd.get_dummies(df_test)

    train_labels = df_train['TARGET']
    
    # Align the training and testing data, keep only columns present in both dataframes
    df_train, df_test = df_train.align(df_test, join = 'inner', axis = 1)
    
    # Add the target back in
    df_train['TARGET'] = train_labels
    
    print('Training Features shape: ', df_train.shape)
    print('Testing Features shape: ' , df_test.shape)

    return df_train,df_test

#    # Private encoding I implemented before I was familiar with the above, don't use unless there's a strong motivation         
#    else:
#        if rebuildEncoders:
#            dictpath = "dictionaries"
##            os.rm(dictpath+'/*txt')   ## TODO: fix this with correct syntax, want to remove old encoders if this option is specified
#            for c in stringcols:
#
#                # Name of dictionary file
#                dictname = dictpath+"/"+c+".txt"
#
#                if os.path.exists(dictname):
#                    with open(dictname) as f:
#                         c_dict = json.load(f)
#                else:
#                    c_dict = {}
#                    c_dict[NAsub] = NAsub
#
#                if debug:
#                        print("Building dictionary for column {0} with example values:\n{1}".format(c, newdf_train[[c]].head()))
#                c_dict = makeNameTokens([newdf_train, newdf_test], c, c_dict) # append dictionaries for each column
#                # write the dictionary to file
#                with open(dictname, 'w') as file:
#                     file.write(json.dumps(c_dict))
#
#        # Replace string columns with tokens
#        for c in stringcols:
#            # Name of dictionary file
#            dictname = dictpath+"/"+c+".txt"
#
#            if os.path.exists(dictname):
#                with open(dictname) as f:
#                     c_dict = json.load(f)
#
#            df_train = df_train.replace({c: c_dict})
#            df_test  = df_test.replace({c: c_dict})



def removeNAcols(df_train,df_test, maxNAfrac=0.2):
    # Remove features that are mostly NA (for now).
    rareCols = []
    for c in df_train.columns.tolist():
        if (df_train.loc[df_train[c] == NAsub]).shape[0] > maxNAfrac:
           rareCols.append(c)

    # Leave the target column untouched
    if targetCol in rareCols: rareCols.remove(targetCol)

    print("\nGoing to remove the following columns due to high fraction of NA:{}".format(rareCols))

    df_train = df_train.drop(columns=rareCols)
    df_test = df_test.drop(columns=rareCols)

    return df_train,df_test


def engineerFeatures(df):
    df['EXT_SOURCE_SUM'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis=1)     
    df['EXT_SOURCE_SUM12'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis=1)     
    df['EXT_SOURCE_SUM23'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis=1)     
    df['EXT_SOURCE_PRODUCT'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].product(axis=1)

    df['EXT_SOURCE_12'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2']].product(axis=1)
    df['EXT_SOURCE_13'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_3']].product(axis=1)
    df['EXT_SOURCE_23'] = df[['EXT_SOURCE_2', 'EXT_SOURCE_3']].product(axis=1)

    df['EXT_SOURCE_1div2'] = df['EXT_SOURCE_1'].divide(df['EXT_SOURCE_2'].replace(0,np.nan)).fillna(0)
    df['EXT_SOURCE_1div3'] = df['EXT_SOURCE_1'].divide(df['EXT_SOURCE_3'].replace(0,np.nan)).fillna(0)
    df['EXT_SOURCE_2div3'] = df['EXT_SOURCE_2'].divide(df['EXT_SOURCE_3'].replace(0,np.nan)).fillna(0)
    df['BALANCE_PER_UNEMPLOYMENT'] = df['AMT_BALANCE'].divide(df['DAYS_EMPLOYED'].replace(0,np.nan)).fillna(0)

    df['EXT_SOURCE_1_sq'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_1']].product(axis=1) 
    df['EXT_SOURCE_2_sq'] = df[['EXT_SOURCE_2', 'EXT_SOURCE_2']].product(axis=1) 
    df['EXT_SOURCE_3_sq'] = df[['EXT_SOURCE_3', 'EXT_SOURCE_3']].product(axis=1) 
    df['EXT_SOURCE_1_cu'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_1', 'EXT_SOURCE_1']].product(axis=1) 
    df['EXT_SOURCE_2_cu'] = df[['EXT_SOURCE_2', 'EXT_SOURCE_2', 'EXT_SOURCE_2']].product(axis=1) 
    df['EXT_SOURCE_3_cu'] = df[['EXT_SOURCE_3', 'EXT_SOURCE_3', 'EXT_SOURCE_3']].product(axis=1) 
    
    return df 


def preprocess(df_train, df_test, buildDictionaries, doFeatureScaling, noRareFeatures, makeNewFeatures=True, myLabels=False, targetCol='TARGET'):

    if debug:
        print("\npreprocess (beginning): Number of entries missing in train data: \n{0}".format(df_train.isnull().sum()))
        print("\npreprocess (beginning): Number of entries missing in test data: \n{0}".format(df_test.isnull().sum()))

    # Remove features that are mostly NA (for now).
    if noRareFeatures:
        newdf_train, newdf_test = removeNAcols(newdf_train,newdf_test, maxNAFrac=0.3)

    # Let's start with an ultrasimple na replace
#    newdf_train = df_train.fillna(df_train.mean())
#    newdf_test = df_test.fillna(df_test.mean())
    newdf_train = df_train.fillna(0)
    newdf_test = df_test.fillna(0)

    # Add potentially useful variables
    if makeNewFeatures:
        newdf_train = engineerFeatures(newdf_train)
        newdf_test = engineerFeatures(newdf_test)

    if debug:
        dtypeCount_train =[newdf_train.iloc[:,i].apply(type).value_counts() for i in range(newdf_train.shape[1])]
        dtypeCount_test =[newdf_test.iloc[:,i].apply(type).value_counts() for i in range(newdf_test.shape[1])]
        print(dtypeCount_train)
        print(dtypeCount_test)

    stringcols = []
    numericcols = newdf_train.columns.tolist()
    for c in newdf_train.columns[newdf_train.dtypes=='object']:
        stringcols.append(c)
        numericcols.remove(c)
    for c in newdf_test.columns[newdf_test.dtypes=='object']:
        if c not in stringcols:
            stringcols.append(c)
            numericcols.remove(c)
  
    print("string cols:\n{0}".format(stringcols))
    print("numeric cols:\n{0}".format(numericcols))

    if doFeatureScaling:
        newdf_train, newdf_test = featureScaler(newdf_train, newdf_test, stringcols, numericcols, targetCol)

    for c in newdf_train.filter(regex='SK_ID*').columns:
        print("in rare features block, found SK column {}".format(c))

    if debug:
        for c in newdf_train.columns["_y" in newdf_train.columns or "_x" in newdf_train.columns]:
            print ("WARNING (preprocess): Found sister column {0}".format(c))

    # Encode object columns
    newdf_train,newdf_test = encode_dfs(newdf_train, newdf_test, stringcols, numericcols, myEncoding = myLabels)

    print("\nNumber of columns in train and test data, respectively: {},   {}".format(len(newdf_train.columns.tolist()), len(newdf_test.columns.tolist())))

    print("\nMost positive correlations with TARGET column in training dataset:\n{}".format(newdf_train.corr()['TARGET'].sort_values(ascending=False).head(10)))
    print("\nMost negative correlations with TARGET column in training dataset:\n{}".format(newdf_train.corr()['TARGET'].sort_values(ascending=False).tail(10)))

    if debug:
        print("\npreprocess (end): Number of entries missing in train data: \n{0}".format(newdf_train.isnull().sum()))
        print("\npreprocess (end): Number of entries missing in test data: \n{0}".format(newdf_test.isnull().sum()))

    return newdf_train, newdf_test
