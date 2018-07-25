
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

def Scale(df, y='TARGET'):

    if debug:
        print("Doing StandardScaler")

#    print("\nInside Scale(): # of entries missing in data: \n{0}".format(df.isnull().sum()))

    idxcol = []
    for c in df.filter(regex='SK_ID*').columns:
        idxcol.append(c)

    # Don't scale target column!!
    if y in df.columns:
        print("Found {} in columns, removing it before".format(y))
        idxcol.append(y)
        print("mean of target column before handling: {0}".format(df[y].mean()))

    print idxcol
    targetdf = pd.DataFrame(df[idxcol], columns=idxcol)

    df = df.drop(columns=idxcol)
    xcol = df.columns
    print("\nInside Scale(): # rows in df before scale fit: {0}".format(df.shape))
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

    print ("before merge scaleddf and targetdf")
    print("SK_ in df? {}".format( [c for c in df.filter(regex='SK_ID*').columns]) )
    print("TARGET in df? {}".format( 'TARGET' in df.columns.tolist() ) )
    scaleddf = pd.concat( [targetdf,scaleddf], axis=1 )
#    scaleddf = targetdf.merge( scaleddf, on='SK_ID_CURR' )
    print ("after merge scaleddf and targetdf")
    print("SK_ in scaleddf? {}".format( [c for c in scaleddf.filter(regex='SK_ID*').columns]) )
    print("TARGET in scaleddf? {}".format( 'TARGET' in scaleddf.columns.tolist() ) )


#    print("mean of target column before handling: {0}".format(df[y].mean()))

#    if debug:
#    print("\nInside Scale(): # rows in targetdf: {0}".format(targetdf.shape))
    print("\nInside Scale(): # rows in df: {0}".format(df.shape))
    print("\nInside Scale(): # rows in scaleddf: {0}".format(scaleddf.shape))
    print("SK_ columns are:\n{0}".format( [ c for c in scaleddf.columns.tolist() if "SK_" in c ] ) )

    print("End of StandardScaler")
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


def preprocess(df_train, df_test, buildDictionaries, doFeatureScaling, tokenizeNames, noRareFeatures, targetCol='TARGET'):

    # test and xval df's are smaller, will run out of rows for high skipranges
    #if df.shape[0] == 0:
    #    return df

    if debug:
        print("\npreprocess (beginning): Number of entries missing in train data: \n{0}".format(df_train.isnull().sum()))
        print("\npreprocess (beginning): Number of entries missing in test data: \n{0}".format(df_test.isnull().sum()))

    # Let's start with an ultrasimple na replace
    NAsub = -11
    newdf_train = df_train.fillna(NAsub)
    newdf_test = df_test.fillna(NAsub)
    print("Column {0} with example values:\n{1}".format('NAME_CONTRACT_TYPE', newdf_train[['NAME_CONTRACT_TYPE']].head()))

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
  
    # Leave the target column untouched
#    if targetCol in stringcols: stringcols.remove(targetCol)

    print("string cols:\n{0}".format(stringcols))
    print("numeric cols:\n{0}".format(numericcols))

    # Scale variables after strings have been converted
    print("Length of numericcols: {}".format(len(numericcols)))
    print("is targetCol in numericcols?: {}".format(targetCol in numericcols))

    if doFeatureScaling:
        newdf_train_string = pd.DataFrame(newdf_train[stringcols], columns=stringcols)
        newdf_train_num    = pd.DataFrame(newdf_train[numericcols], columns=numericcols)
        newdf_train_num, scalefit, cols_in_scalefit = Scale(newdf_train[numericcols])  
        newdf_train = pd.concat([newdf_train_num,newdf_train_string], axis=1)

        # Now use the scale fit above for the test data (remember test data doesn't have targetCol):
        if targetCol in numericcols: numericcols.remove(targetCol)
        # there may be other columns that are present that the fit doesn't use:
        cols_dont_scale = [x for x in numericcols if x not in cols_in_scalefit]
        newdf_test_string = pd.DataFrame(newdf_test[stringcols], columns=stringcols)
        newdf_test_num    = pd.DataFrame(newdf_test[numericcols], columns=numericcols)
        newdf_test_temp = pd.DataFrame(newdf_test[cols_dont_scale], columns=cols_dont_scale) 
        print("Going to remove {} from {}".format(cols_dont_scale,numericcols))
        for c in cols_dont_scale:
            numericcols.remove(c)
        if newdf_test.shape[0] > 0:
            newdf_test_num[numericcols] = scalefit.transform(newdf_test[numericcols])
        newdf_test = pd.concat([newdf_test_num,newdf_test_string], axis=1)

    for c in newdf_train.filter(regex='SK_ID*').columns:
        print("in rare features block, found SK column {}".format(c))

    if debug:
        for c in newdf_train.columns["_y" in newdf_train.columns or "_x" in newdf_train.columns]:
            print ("preprocess: Found sister column {0}".format(c))

    dictpath = "dictionaries"
    if buildDictionaries:
        for c in stringcols:
    
            # Name of dictionary file
            dictname = dictpath+"/"+c+".txt"
    
            if os.path.exists(dictname):
                with open(dictname) as f:
                     c_dict = json.load(f)
            else:
                c_dict = {}
                c_dict[NAsub] = NAsub
    
            if debug:
                    print("Building dictionary for column {0} with example values:\n{1}".format(c, newdf_train[[c]].head()))
            c_dict = makeNameTokens([newdf_train, newdf_test], c, c_dict) # append dictionaries for each column
            # write the dictionary to file
            with open(dictname, 'w') as file:
                 file.write(json.dumps(c_dict))


    if tokenizeNames:
        # Replace string columns with tokens
        for c in stringcols:
            # Name of dictionary file
            dictname = dictpath+"/"+c+".txt"
    
            if os.path.exists(dictname):
                with open(dictname) as f:
                     c_dict = json.load(f)

            newdf_train = newdf_train.replace({c: c_dict})
            newdf_test = newdf_test.replace({c: c_dict})
#    else:
#        # OR simply label string columns
#
#        print(newdf['FONDKAPREMONT_MODE'].unique())
#        print(newdf[['FONDKAPREMONT_MODE']].head(20))
#        clfs = {c:LabelEncoder() for c in stringcols}
#        for col, clf in clfs.items():
#            newdf[col] = clf.fit_transform(newdf[col])
#
#        oh_encs = {c:OneHotEncoder() for c in stringcols}
#        for col, oh_enc in oh_encs.items():
#            if not np.all(np.isfinite(newdf[col])):
#                print("Found non-finite value for column {0}".format(col))
#
#                print(newdf['FONDKAPREMONT_MODE'].unique())
#                print(newdf.loc[np.isnan(newdf['FONDKAPREMONT_MODE']), 'FONDKAPREMONT_MODE'])
#                print(newdf[['FONDKAPREMONT_MODE']].head())
#
#	    print("Are any entries NA? {0}".format(np.any(np.isnan(newdf[col]))) )
#	    print("Are all entries finite? {0}".format(np.all(np.isfinite(newdf[col]))))
#            X = oh_enc.fit_transform(newdf[col].values.reshape(-1,1))
#            X = X.toarray()
#            print("col={0}, oh_enc={1}".format(col, oh_enc))
#            print("Values of column {0}".format(newdf[col].values))
#            print("X.shape = {0}".format(X.shape))
#            dfOneHot = pd.DataFrame(X, columns = [ col+"_"+str(int(i)) for i in range(X.shape[1])] )
#            newdf = pd.concat([newdf, dfOneHot], axis=1)

#    if debug:
#        print("After tokenizing")
#        dtypeCount_x =[newdf_train.iloc[:,i].apply(type).value_counts() for i in range(newdf_train.shape[1])]
#        print(dtypeCount_x)

    # Remove features that are mostly NA (for now).
    if noRareFeatures:
        for c in newdf_train.filter(regex='SK_ID*').columns:
            print("in rare features block, found SK column {}".format(c))
        rareCols = []
        # Define what you mean by 'rare'
        maxNAfrac = 0.25*newdf_train.shape[0]
        for c in newdf_train.columns.tolist():
            if debug:
                print("Column {}, Head of it:\n{}".format(c,newdf_train[[c]].head()))
            if (newdf_train.loc[newdf_train[c] == NAsub]).shape[0] > maxNAfrac:
               rareCols.append(c)
        # Leave the target column untouched
        if targetCol in rareCols: rareCols.remove(targetCol)
        
        newdf_train = newdf_train.drop(columns=rareCols)
        newdf_test = newdf_test.drop(columns=rareCols)

    print("\nNumber of columns in train and test data, respectively: {},   {}".format(len(newdf_train.columns.tolist()), len(newdf_test.columns.tolist())))
    if debug:
        print("\npreprocess (end): Number of entries missing in train data: \n{0}".format(newdf_train.isnull().sum()))
        print("\npreprocess (end): Number of entries missing in test data: \n{0}".format(newdf_test.isnull().sum()))

    return newdf_train, newdf_test
