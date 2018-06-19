
# This will be useful to convert letter to ints
from string import ascii_lowercase, ascii_uppercase

# For mean/dev scaling. Important for PCA, and good practice in general
from sklearn.preprocessing import StandardScaler

import pandas as pd


debug = True


def Scale(df, y='TARGET'):

    if debug:
        print("Doing StandardScaler")

    scaler = StandardScaler()

    # Don't scale target column!!
    keepTarget = False
    if y in df.columns:
        print("mean of target column before handling: {0}".format(df[y].mean()))
        keepTarget = True
        targetdf = df[y]
        df = df.drop(columns=[y])

    col_name = df.columns

    scaleddf = scaler.fit_transform(df)
    scaleddf = pd.DataFrame(scaleddf, columns = col_name)

    if keepTarget:
        scaleddf[y] = targetdf 
        print("mean of target column after handling: {0}".format(scaleddf[y].mean()))


    return scaleddf


def makeNameTokens(df, col):
        name_dict = {}

        # Useful for NA's
        name_dict[0] = 0

        numbervalue = 1
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
    cidx = df.columns.get_loc(col)

    newcol = df.apply(lambda row: namedict[ row[col] ], axis=1)

    return newcol


def preprocess(df):

    # Let's start with an ultrasimple na replace
    newdf = df.fillna(0)
    if debug:
        print("This dataset has the following datatypes:\n{0}\n".format(newdf.dtypes))

        # Some of the Fare entries are strings. 
        dtypeCount_x =[df.iloc[:,i].apply(type).value_counts() for i in range(df.shape[1])]
        print(dtypeCount_x)


    stringcols = []
    for c in newdf.columns[newdf.dtypes=='object']:
        stringcols.append(c)
    print("Columns with non-numerical values are:\n{0}".format(stringcols))

    for c in stringcols:
        if debug:
                print("Building dictionary for column {0} with example values:\n{1}".format(c, newdf[c].head()))
        tempdict = makeNameTokens(newdf,c)
        newdf[c] = tokenizeNames(newdf,c,tempdict)

    newdf = Scale(newdf)

    return newdf
