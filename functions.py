import numpy as np  
import pandas as pd



def create_column_filter(df):
    df = df.copy()
    for i, col in enumerate(df):
        if col != "CLASS" and col != "ID":
            # array of unique (not nan) values
            col_unique_values = df[col].dropna().unique()
            # drop column of there is <= 1 value in that column
            if len(col_unique_values) <= 1:
                df = df.drop(columns=col)
    # get all remaining column names
    column_filter = df.columns
    return df, column_filter


def apply_column_filter(df, column_filter):
    df = df.copy()
    df = df.drop(columns=df.columns.difference(column_filter))
    return df


def create_normalization(df, normalizationtype="minmax"):
    df = df.copy()
    normalization = {}
    # minmax normalization
    if normalizationtype == "minmax":
        for i, col in enumerate(df):
            # only change column that are not CLASS or ID and of type int or float
            if col != "CLASS" and col != "ID":
                if df[col].dtype == np.int64 or df[col].dtype == np.float64:
                    # x' = (x - min)/(max - min)
                    min = df[col].min()
                    max = df[col].max()
                    df[col] = [(x-min)/(max-min) for x in df[col]]
                    # add information to normalization dictionary
                    normalization[col] = ("minmax", min, max)
    # zscore normalization
    elif normalizationtype == "zscore":
        for i, col in enumerate(df):
            # only change column that are not CLASS or ID and of type int or float
            if col != "CLASS" and col != "ID":
                if df[col].dtype == np.int64 or df[col].dtype == np.float64:
                    # x' = (x - mean)/std
                    mean = df[col].mean()
                    std = df[col].std()
                    df[col] = df[col].apply(lambda x: (x-mean)/std)
                    # add information to normalization dictionary
                    normalization[col] = ("zscore", mean, std)
    return df, normalization


def apply_normalization(df, normalization):
    df = df.copy()
    for i, col in enumerate(df):
        if col in normalization:
            if normalization[col][0] == "minmax":
                # x' = (x - min)/(max - min)
                min = normalization[col][1]
                max = normalization[col][2]
                df[col] = [(x-min)/(max-min) for x in df[col]]
            elif normalization[col][0] == "zscore":
                # x' = (x - mean)/std
                mean = normalization[col][1]
                std = normalization[col][2]
                df[col] = df[col].apply(lambda x: (x-mean)/std)
    return df


def create_imputation(df):
    df = df.copy()
    imputation = {}
    for i, col in enumerate(df):
        # only change column that are not CLASS or ID and of type int or float
        if col != "CLASS" and col != "ID":
            # numeric
            if df[col].dtype == np.int64 or df[col].dtype == np.float64:
                # replace nan with mean
                mean = df[col].mean()
                # if column is nan, then mean should be 0
                if np.isnan(mean):
                    mean = 0
                df[col].fillna(mean, inplace=True)
                # add to imputation
                imputation[col] = mean
            # categoric
            elif df[col].dtype == object or df[col].dtype == "category":
                # replace nan with mode
                mode = df[col].mode()[0]
                # if column is na, then replace with "" or cat.categories[0]
                if pd.isna(mode):
                    if df[col].dtype == object:
                        mode = ""
                    elif df[col].dtype == "category":
                        mode = cat.categories[0]
                df[col].fillna(mode, inplace=True)
                # add to imputation
                imputation[col] = mode
    return df, imputation


def apply_imputation(df, imputation):
    df = df.copy()
    for i, col in enumerate(df):
        if col in imputation:
            # numeric
            if df[col].dtype == np.int64 or df[col].dtype == np.float64:
                df[col].fillna(imputation[col], inplace=True)
            # categoric
            elif df[col].dtype == object or df[col].dtype == "category":
                df[col].fillna(imputation[col], inplace=True)
    return df


def create_bins(df, nobins=10, bintype="equal-width"):
    df = df.copy()
    binning = {}
    for i, col in enumerate(df):
        # only change column that are not CLASS or ID and of type int or float
        if col != "CLASS" and col != "ID":
            # numeric
            if df[col].dtype == np.int64 or df[col].dtype == np.float64:
                # equal width
                if bintype == "equal-width":
                    res, bins = pd.cut(
                        df[col], nobins, retbins=True, labels=list(range(0, nobins)))
                    df[col] = res
                    # add to binning
                    binning[col] = bins
                    # change first and last element, so that everything fits in the range
                    binning[col][0] = -np.inf
                    binning[col][-1] = np.inf
                # equal size
                elif bintype == "equal-size":
                    res, bins = pd.qcut(
                        df[col], nobins, retbins=True, duplicates='drop')
                    df[col] = res
                    # add to binning
                    binning[col] = bins
                    # change first and last element, so that everything fits in the range
                    binning[col][0] = -np.inf
                    binning[col][-1] = np.inf
    # all columns to type category
    df = df.astype("category")
    return df, binning


def apply_bins(df, binning):
    df = df.copy()
    for i, col in enumerate(df):
        if col in binning:
            df[col] = pd.cut(df[col], binning[col], labels=list(
                range(0, len(binning[col])-1)))
     # all columns to type category
    df = df.astype("category")
    return df


def create_one_hot(df):
    # modify the copy (the input dataframe should be kept unchanged)
    df_new = df.copy()
    one_hot = {}
    for col in df.columns:
        # only consider the columns of type "object" or "category"
        if (df_new[col].dtype == 'object' or df_new[col].dtype == 'category'):
            # and only those not labeled "CLASS" or "ID"
            if (col != 'CLASS' and col != 'ID'):
                df_new[col] = df_new[col].astype('category')
                one_hot[col] = df_new[col].cat.categories
                # convert categorical variable into dummy/indicator variables.
                df_onehot = pd.get_dummies(df_new[col])
                # remove original categoric feature
                df_new = df_new.drop(columns=col, axis=1)
                df_new = pd.concat([df_new, df_onehot], axis=1)
    return df_new, one_hot


def apply_one_hot(df, one_hot):
    # modify the copy (the input dataframe should be kept unchanged)
    df_new = df.copy()
    for col in df.columns:
        if col in one_hot.keys():
            for i in one_hot[col]:
                # create new column names by merging the original column name and the categorical value
                name_col = col + '-' + i
                # all new columns to be of type "float"
                new_col = pd.Series((df[col] == i).astype('float'))
                df_new[name_col] = new_col
            df_new = df_new.drop(columns=col, axis=1)
    return df_new


def split(df, testfraction=0.5):
    df_new = df.copy()
    # prefix corresponds to the test instances, and the suffix to the training instances
    index = np.random.permutation(df_new.index)
    length = df_new.shape[0]
    testlength = int(testfraction*length)
    # test instances
    test_index = index[:testlength]
    # training instances
    train_index = index[testlength:]
    trainingdf = df_new.iloc[train_index, :]
    testdf = df_new.iloc[test_index, :]
    return trainingdf, testdf


def accuracy(df, correctlabels):
    N = len(df)
    labels = df.idxmax(axis=1)
    predictions = (labels == correctlabels).sum(axis=0)
    accuracy = predictions/N
    return accuracy


def folds(df, nofolds=10):
    df_new = df.copy()
    folds = []
    # prefix corresponds to the test instances, and the suffix to the training instances
    index = np.random.permutation(df_new.index)
    for i in range(nofolds):
        # test instances
        test_index = len(df_new)*(i+1)/nofolds
        # training instances
        train_index = len(df_new)*i/nofolds
        folds.append(df_new[int(train_index): int(test_index)])
    return folds


def brier_score(df, correctlabels):
    df_new = df.copy()
    label_array = []

    for i in range(len(df_new.columns)):
        class_vector = []
        sqr_value = 0
        for value in correctlabels:
            if value == df_new.columns[i]:
                class_vector.append(1)
            else:
                class_vector.append(0)

        label_array.append(class_vector)
    label_array = np.array(label_array)
    df_new = df_new.to_numpy()
    df_new = df_new.transpose()
    sqr_value = np.subtract(label_array, df_new)**2
    return np.sum(sqr_value.mean(axis=1))


def auc(df, correctlabels):

    aucs = []

    for col in df.columns:
        score = {}
        values = df[col].values
        val_dec = sorted(values, reverse=True)  # hint 4, reverse ordering

        for elem in val_dec:
            elems = elem
            tp = 0
            fp = 0

            for i in range(len(values)):
                if values[i] == elems:
                    label = col
                    if label == correctlabels[i]:
                        tp = tp + 1
                    else:
                        fp = fp + 1

            score[elems] = [tp, fp]

        AUC = 0
        Cov_tp = 0
        Tot_tp = 0
        Tot_fp = 0
        for i in score:
            Tot_tp += score[i][0]
            Tot_fp += score[i][1]

        for j in score:
            fp = score[j][1]
            tp = score[j][0]
            if fp == 0:
                Cov_tp += tp
            elif tp == 0:
                AUC += (Cov_tp/Tot_tp)*(fp/Tot_fp)
            else:
                AUC += (Cov_tp/Tot_tp)*(fp/Tot_fp)+(tp/Tot_tp)*(fp/Tot_fp)/2
                Cov_tp += tp
        aucs.append(AUC)

    freqs = []

    for col in df.columns:
        freq = correctlabels.value_counts()[col]
        freq = freq /len(correctlabels)
        freqs.append(freq)

    auc_final = np.dot(freqs, aucs)
    return auc_final