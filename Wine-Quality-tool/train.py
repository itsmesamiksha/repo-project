import matplotlib.pyplot as plt
# Packages Part
import pandas as pd
import numpy as np
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
#get_ipython().run_line_magic('matplotlib', 'inline')
print("data load started ")

wineQuality_data='/workspaces/repo-project/Wine-Quality-tool/winequality-red.csv'
print("data load completed ")
wine_df= pd.read_csv(wineQuality_data)
#wine_df.head(2)
#print("Data understanding started")
wine_df.count()
#print("no of records :" , wine_df.count())
wine_df.info()
#print("Null in dataset check in progress ")
wine_df.isnull().sum()
def column_name_standard(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df
column_name_standard(wine_df)
wine_df['good_quality'] = [1 if q >= 7 else 0 for q in wine_df['quality']]
wine_df.drop('quality', axis=1, inplace=True)
#parameter for requires seed
random_value = 24
def data_prep(wine_df):
    main_copy = wine_df.copy()
    #print(main_copy['good_quality'].value_counts())
    xf = main_copy.columns
    X = main_copy.drop(['good_quality'],axis=1)
    y = main_copy['good_quality']
    # 20% test, 60% train, 20% validation
    X_temp, X_test, y_temp, y_test = train_test_split(    X, y, test_size=0.2, random_state=random_value, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(    X_temp, y_temp, test_size=0.25, random_state=random_value, stratify=y_temp)
    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)
    sm = SMOTE(random_state=random_value)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print("Train shapes after SMOTE:", X_train_res.shape, y_train_res.shape)
    scaler = StandardScaler()
    X_train_res_scaled = scaler.fit_transform(X_train_res)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return (X_train_res_scaled, y_train_res,
        X_val_scaled, y_val,
        X_test_scaled, y_test)


def train(wine_df,best_depth,n_estimators):
    X_train_res_scaled, y_train_res, X_val_scaled, y_val, X_test_scaled, y_test = data_prep(wine_df)
    model = xgb.XGBClassifier(random_state=random_value,
                                      max_depth=best_depth, n_estimators=100,
                                     use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train_res_scaled, y_train_res)
    return model,X_train_res_scaled, y_train_res, X_val_scaled, y_val, X_test_scaled, y_test

def predict(model, X,y, name="Set"):
    y_pred_prob = model.predict_proba(X)[:, 1]
    y_pred_class = model.predict(X)
    auc = roc_auc_score(y, y_pred_prob)
    acc = accuracy_score(y, y_pred_class)
    print(f"\n{name} Evaluation:")
    print(f"AUC:  {auc:.4f}")
    print(f"ACC:  {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred_class))
    print("\nClassification Report:")
    print(classification_report(y, y_pred_class))
    return y_pred_class


# Traning and testing the the model
best_depth = 5
n_estimators=100
model, X_train_res_scaled, y_train_res, X_val_scaled, y_val, X_test_scaled, y_test = train(
    wine_df, best_depth=5, n_estimators=100
)
predict(model, X_train_res_scaled, y_train_res, "TRAIN")
predict(model, X_val_scaled, y_val, "VALIDATION")
predict(model, X_test_scaled, y_test, "TEST")


print('-----------------------------------')

output_file = f'model.bin'

with open(output_file, 'wb') as f_out:

    pickle.dump(model, f_out)

#print(f'The model is saved to {output_file}')
# Evaluate
#predict(final_xgmodel, X_val, y_val, "Validation")
#predict(final_xgmodel, X_test, y_test, "Test")

# ---------------------------------------------------------
# 10. EVALUATE ON TRAIN/VAL/TEST
# ---------------------------------------------------------
#predict(model, X_train_res_scaled, y_train_res, "TRAIN")
#predict(model, X_val_scaled, y_val, "VALIDATION")
#predict(model, X_test_scaled, y_test, "TEST")



