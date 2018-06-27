
# This will be useful to convert letter to ints
from string import ascii_lowercase, ascii_uppercase

# For mean/dev scaling. Important for PCA, and good practice in general
from sklearn.preprocessing import StandardScaler

import os

from collections import OrderedDict

import pandas as pd

import json

debug = True

def Scale(df, y='TARGET'):

    if debug:
        print("Doing StandardScaler")

#    print("\nInside Scale(): # of entries missing in data: \n{0}".format(df.isnull().sum()))
    scaler = StandardScaler()

    skcol = []
    for c in df.filter(regex='SK_ID*').columns:
        skcol.append(c)

    # Don't scale target column!!
    keepTarget = False
    if y in df.columns:
        print("mean of target column before handling: {0}".format(df[y].mean()))
        keepTarget = True
        skcol.append(y)

    targetdf = df[skcol].copy()

    df = df.drop(columns=skcol)
    xcol = df.columns

    scaleddf = scaler.fit_transform(df)
    scaleddf = pd.DataFrame(scaleddf, columns = xcol)
    scaleddf['SK_ID_CURR'] = targetdf['SK_ID_CURR'].values 

    scaleddf = scaleddf.merge( targetdf, on='SK_ID_CURR' )
    if debug:
        print("\nInside Scale(): # rows in targetdf: {0}".format(targetdf.shape))
        print("\nInside Scale(): # rows in df: {0}".format(df.shape))
        print("\nInside Scale(): # rows in scaleddf: {0}".format(scaleddf.shape))

    return scaleddf


def makeNameTokens(df, col, name_dict):

        numbervalue = name_dict.values()[-1] + 1
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


def preprocess(df, buildDictionaries):

    if debug:
        print("\npreprocess (beginning): Number of entries missing in data: \n{0}".format(df.isnull().sum()))

    # Let's start with an ultrasimple na replace
    newdf = df.fillna(0)
    if debug:
        dtypeCount_x =[newdf.iloc[:,i].apply(type).value_counts() for i in range(newdf.shape[1])]
        print(dtypeCount_x)

    stringcols = []
    for c in newdf.columns[newdf.dtypes=='object']:
        stringcols.append(c)


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
                c_dict[0] = 0
    
            if debug:
                    print("Building dictionary for column {0} with example values:\n{1}".format(c, newdf[c].head()))
    
            c_dict = makeNameTokens(newdf, c, c_dict) # append dictionaries for each column
    
            # write the dictionary to file
            with open(dictname, 'w') as file:
                 file.write(json.dumps(c_dict))

    # Now replace string columns with tokens
    for c in stringcols:
        # Name of dictionary file
        dictname = dictpath+"/"+c+".txt"

        if os.path.exists(dictname):
            with open(dictname) as f:
                 c_dict = json.load(f)

        newdf = newdf.replace({c: c_dict})

    if debug:
        print("After tokenizing")
        dtypeCount_x =[newdf.iloc[:,i].apply(type).value_counts() for i in range(newdf.shape[1])]
        print(dtypeCount_x)

    # Scale variables after strings have been converted
    newdf = Scale(newdf)

    if debug:
        print("\npreprocess (end): Number of entries missing in data: \n{0}".format(newdf.isnull().sum()))

    return newdf
