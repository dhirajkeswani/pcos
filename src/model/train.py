# Import libraries

import argparse
import glob
import os

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

# define functions
def main(args):
    # TO DO: enable autologging
    mlflow.autolog()
    

    # read data
    df = get_csvs_df(args.training_data)

    preprocessed_df = preprocess_data(df)

    # split data
    X_train, X_test, y_train, y_test = split_data(preprocessed_df)

    # train model
    train_model(args.model, X_train, X_test, y_train, y_test)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)

def preprocess_data(df):
    df.drop(['Sl. No','Patient File No.','Unnamed: 44'], axis=1, inplace=True)
    df['AMH(ng/mL)'].replace({'a':1},inplace=True)
    df['II    beta-HCG(mIU/mL)'].replace({'1.99.':1.99},inplace=True)
    imputer = KNNImputer(n_neighbors=23)
    imputed_df = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)
    imputed_df['Fast food (Y/N)'] = np.where(imputed_df['Fast food (Y/N)']>0.5,1,0)
    imputed_df['Marraige Status (Yrs)'] = np.around(imputed_df['Marraige Status (Yrs)'],1)
    return imputed_df

# TO DO: add function to split data
def split_data(df):
    X = df.drop('PCOS (Y/N)',axis=1)
    y = df['PCOS (Y/N)']

    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=3,stratify=y)
    return (X_train, X_test, y_train, y_test)


# def train_model(reg_rate, X_train, X_test, y_train, y_test):
#     # train model
#     LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)

def train_model(model, X_train, X_test, y_train, y_test):  #we can take another param as model_params as a list and apply that
    # train model
    if model == "RandomForestClassifier":
        selected_model = RandomForestClassifier()
        selected_model.fit(X_train, y_train)

    elif model == "AdaBoostClassifier":
        selected_model = AdaBoostClassifier()
        selected_model.fit(X_train, y_train)

    elif model == "LogisticRegression":
        selected_model = LogisticRegression()
        selected_model.fit(X_train, y_train)

    else:
        raise RuntimeError("Valid model name not selected")

    y_train_pred = selected_model.predict(X_train)
    y_test_pred = selected_model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("model", model)
    mlflow.autolog()



def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--model", dest='model',
                        type=str, default="LogisticRegression")

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
