import os
import warnings
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
import joblib
import argparse
from get_data import read_params
import json


def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"] 
    model_dir = config["model_dir"]
    
    learning_rate = config['estimators']['XGBClassifier']['learning_rate']
    max_depth = config['estimators']['XGBClassifier']['max_depth']
    n_estimators = config['estimators']['XGBClassifier']['n_estimators'] 
    min_child_weight = config['estimators']['XGBClassifier']['min_child_weight'] 
    
    target = [config['base']['target_col']]
    
    train = pd.read_csv(train_data_path, sep=',', encoding='utf-8')
    test = pd.read_csv(test_data_path, sep=',', encoding='utf-8')
    
    y_train = train[target]
    y_test = test[target]
    
    X_train = train.drop(target, axis=1)
    X_test = test.drop(target, axis=1)
    
    clf = xgb.XGBClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        min_child_weight=min_child_weight
    )
    clf.fit(X_train, y_train)
    
    predicted = clf.predict(X_test)
    
    score = f1_score(y_test, predicted)
    
    print(f"XGBClassifier model f1_score : {score}")
    
    ########################################
    
    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]
    
    with open(scores_file, 'w') as f:
        scores_ = {
            'f1_score': score,
        }
        json.dump(scores_, f, indent=4)
        
    with open(params_file, 'w') as f:
        params_ = {
            'learning_rate': learning_rate,
            'max_depth':max_depth,
            'n_estimators': n_estimators,
            'min_child_weight': min_child_weight
        }
        json.dump(params_, f, indent=4)

    #########################################
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(clf, model_path)
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default='params.yaml')
    parsed_args = args.parse_args()
    train_and_evaluate(parsed_args.config)
    