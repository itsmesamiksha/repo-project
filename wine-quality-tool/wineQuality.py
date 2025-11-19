#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Packages Part
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import xgboost as xgb
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


wineQuality_data='C:/Users/samik/Documents/pythonlearn/MLZC2025/ML-ZC-2025/WineQuality/winequality-red.csv'


# In[3]:


wine_df= pd.read_csv(wineQuality_data)


# In[4]:


wine_df.head(2)


# # Data understanding 

# In[5]:


wine_df.count()


# In[6]:


wine_df.info()


# In[7]:


wine_df.isnull().sum()


# # function to standarise column names 

# In[8]:


def column_name_standard(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df


# In[9]:


column_name_standard(wine_df)
wine_df.head(2)


# In[10]:


# Defining a function to calculate importance of features
def mutual_info_all(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    encoded = X.copy()

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            le = LabelEncoder()
            encoded[col] = le.fit_transform(X[col].astype(str))
        else:
            encoded[col] = X[col].fillna(X[col].median())

    mi = mutual_info_regression(encoded, y, discrete_features='auto')

    result = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    return result


# In[11]:


mi_scores = mutual_info_all(wine_df, target='quality')
print(mi_scores)


# # Treat quality column to 0,1 

# In[12]:


print(wine_df['quality'].value_counts())


# In[13]:


sns.countplot(x='quality', data=wine_df, palette='viridis')
plt.title("Wine Quality Distribution")
plt.show()


# In[14]:


wine_df['good_quality'] = [1 if q >= 7 else 0 for q in wine_df['quality']]
wine_df.drop('quality', axis=1, inplace=True)
print(wine_df.head())


# In[15]:


sns.countplot(data = wine_df, x = 'good_quality')
plt.xticks([0,1], ['bad wine','good wine'])
plt.title("Types of Wine")
plt.show()


# In[16]:


xi = wine_df.corrwith(wine_df.good_quality).abs()
xi.sort_values(ascending=False)


# In[17]:


from imblearn.over_sampling import SMOTE


# In[18]:


#parameter for requires seed
random_value = 1000
#Fixing the imbalance using SMOTE Technique
main_copy = wine_df.copy()
print('Original class distribution:')
print(main_copy['good_quality'].value_counts())
xf = main_copy.columns
X = main_copy.drop(['good_quality'],axis=1)
Y = main_copy['good_quality']


# In[19]:


oversample = SMOTE()
X_ros, y_ros = oversample.fit_resample(X, Y)
sns.countplot(x=y_ros)
plt.xticks([0,1], ['bad wine','good wine'])
plt.title("Types of Wine")
plt.show()


# In[20]:


# Step 1: Split off the test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(X_ros, y_ros, test_size=0.2, random_state=random_value)
# Step 2: Split the remaining 80% into train and validation (e.g. 60/20)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=random_value)  


# In[21]:


len(X_train), len(X_val), len(X_test),len(y_train), len(y_val), len(y_test)


# In[22]:


X_train.head(2)
y_train.head(2)


# In[23]:


# Importing LogisticRegression and metrics from sklearn library
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report


# In[24]:


# scale with StandardScaler
scaler = StandardScaler()
# fit to data training
scaler.fit(X_train)
# transform
x_train_scaled = scaler.transform(X_train)
x_val_scaled = scaler.transform(X_val)
x_test_scaled = scaler.transform(X_test)


# # Logistics Regression

# In[25]:


model1 = LogisticRegression(max_iter=1000, random_state=random_value)
model1.fit(x_train_scaled, y_train)


# In[26]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score


# In[27]:


y_val_pred = model1.predict(x_val_scaled)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))


# # testing the model on test data

# In[28]:


print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


# # Logistic regression with cross validation and finding the best C

# In[29]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from tqdm import tqdm


random_value = 42
X_temp, X_test, y_temp, y_test = train_test_split(    X, Y, test_size=0.20, stratify=Y, random_state=random_value)
X_train, X_val, y_train, y_val = train_test_split(    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=random_value)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

sm = SMOTE(random_state=random_value)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Resampled Train:", X_train_res.shape)


scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(X_train_res)
x_val_scaled   = scaler.transform(X_val)
x_test_scaled  = scaler.transform(X_test)


C_values = [0.001, 0.01, 0.1, 0.5, 1, 5, 10]
n_splits = 5

best_C = None
best_auc = -1

print("\n===== CROSS VALIDATION STARTED =====\n")

for C in tqdm(C_values):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_value)
    fold_scores = []

    for train_idx, val_idx in kfold.split(x_train_scaled):

        X_fold_train = x_train_scaled[train_idx]
        X_fold_val   = x_train_scaled[val_idx]

        y_fold_train = y_train_res.iloc[train_idx]
        y_fold_val   = y_train_res.iloc[val_idx]

        model = LogisticRegression(
            max_iter=2000,
            random_state=random_value,
            C=C,
            solver='lbfgs'
        )

        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_fold_val)

        auc = roc_auc_score(y_fold_val, y_pred)
        fold_scores.append(auc)

    mean_auc = np.mean(fold_scores)
    print(f"C={C} → AUC={mean_auc:.4f}")

    if mean_auc > best_auc:
        best_auc = mean_auc
        best_C = C

print("\nBest C value:", best_C)






# In[30]:


final_model = LogisticRegression(
    C=0.5,
    max_iter=2000,
    random_state=random_value
)

final_model.fit(x_train_scaled, y_train_res)


# In[31]:


# ============================================
# 8. VALIDATION EVALUATION
# ============================================
print("\n===== VALIDATION RESULTS =====")
y_val_pred = final_model.predict(x_val_scaled)

print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation AUC:", roc_auc_score(y_val, y_val_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))


# ============================================
# 9. FINAL TEST SET EVALUATION
# ============================================
print("\n===== TEST RESULTS =====")
y_test_pred = final_model.predict(x_test_scaled)

print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test AUC:", roc_auc_score(y_test, y_test_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))


# # random forest classifier with cross validation and finding the best depth

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import roc_auc_score
# from imblearn.over_sampling import SMOTE
# from tqdm import tqdm
# 
# random_value = 42
# X_temp, X_test, y_temp, y_test = train_test_split(    X, Y, test_size=0.20, stratify=Y, random_state=random_value)
# X_train, X_val, y_train, y_val = train_test_split(    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=random_value)
# 
# print("Train:", X_train.shape)
# print("Validation:", X_val.shape)
# print("Test:", X_test.shape)
# 
# sm = SMOTE(random_state=random_value)
# X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
# 
# print("Resampled Train:", X_train_res.shape)
# 
# 
# scaler = StandardScaler()
# 
# x_train_scaled = scaler.fit_transform(X_train_res)
# x_val_scaled   = scaler.transform(X_val)
# x_test_scaled  = scaler.transform(X_test)
# print("Train:", X_train.shape, y_train.shape)
# print("Validation:", X_val.shape, y_val.shape)
# print("Test:", X_test.shape, y_test.shape)
# 
# n_splits = 5
# for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
#     
#     kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_value)
#     scores = []
#     
#     for train_idx, val_idx in kfold.split(x_train_scaled):
#         
#         X_fold_train = x_train_scaled[train_idx]
#         X_fold_val   = x_train_scaled[val_idx]
#         y_fold_train = y_train_res.iloc[train_idx]
#         y_fold_val   = y_train_res.iloc[val_idx]
#         
#         model = LogisticRegression(
#             max_iter=1000,
#             random_state=random_value,
#             C=C,
#             solver='lbfgs'
#         )
# 
#         model.fit(X_fold_train, y_fold_train)
#         y_pred = model.predict(X_fold_val)
#         
#         auc = roc_auc_score(y_fold_val, y_pred)
#         scores.append(auc)
#     
#     print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))
#     

# scores
# #c = 0.01 

# model_LR = LogisticRegression(max_iter=1000, random_state=random_value, C=.5)
# model_LR.fit(x_train_scaled, y_train)
# y_pred = model_LR.predict(x_test_scaled)
# auc = roc_auc_score(y_val, y_pred)

# auc

# y_test_pred = model.predict(x_test_scaled)
# 
# print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# auc

# # Random Forest Classifier

# In[199]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_text
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score


# In[210]:


X_train.shape , X_val.shape ,X_test.shape , X_ros.shape


# In[47]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# 20% test, 60% train, 20% validation
X_temp, X_test, y_temp, y_test = train_test_split(
    X_ros, y_ros, test_size=0.2, random_state=random_value, stratify=y_ros
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=random_value, stratify=y_temp
)
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


# In[48]:


n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_value)

n_estimators_list = [50, 100, 150, 200]
max_depth_list = [2, 4, 6, 8, 10, None]

results = []
best_auc = -1
best_params = None

print("\nStarting hyperparameter search...\n")

for max_depth in max_depth_list:
    for n_estimators in tqdm(n_estimators_list, desc=f"max_depth={max_depth}"):

        fold_aucs = []

        for train_idx, val_idx in kfold.split(X_train):

            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            # Apply SMOTE inside each fold to avoid data leakage
            sm = SMOTE(random_state=random_value)
            X_res, y_res = sm.fit_resample(X_fold_train, y_fold_train)

            # Train Random Forest
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_value,
                n_jobs=-1
            )
            rf.fit(X_res, y_res)

            # Predict probabilities (required for ROC-AUC)
            y_pred = rf.predict_proba(X_fold_val)[:, 1]
            auc = roc_auc_score(y_fold_val, y_pred)
            fold_aucs.append(auc)

        mean_auc = np.mean(fold_aucs)

        print(f"max_depth={max_depth}, n_estimators={n_estimators}, AUC={mean_auc:.4f}")

        # Track best model
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = (max_depth, n_estimators)

        results.append((max_depth, n_estimators, mean_auc))

print("\nBest Hyperparameters:")
print("Max Depth:", best_params[0])
print("Estimators:", best_params[1])
print(f"Best CV AUC: {best_auc:.4f}")


# In[49]:


# ------------------------------------------------------------
# 5. TRAIN FINAL MODEL ON TRAINING + SMOTE
# ------------------------------------------------------------

print("\nTraining final model...")

best_max_depth, best_n_estimators = best_params

sm = SMOTE(random_state=random_value)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

final_rf = RandomForestClassifier(
    max_depth=best_max_depth,
    n_estimators=best_n_estimators,
    random_state=random_value,
    n_jobs=-1
)

final_rf.fit(X_train_res, y_train_res)

# ------------------------------------------------------------
# 6. EVALUATION (VALIDATION + TEST)
# ------------------------------------------------------------

def evaluate(model, X, y, name="Set"):

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

# Evaluate
evaluate(final_rf, X_val, y_val, "Validation")
evaluate(final_rf, X_test, y_test, "Test")


# # Decision Tree Classifier

# In[54]:


# ------------------------------------------------------------
#  DECISION TREE CLASSIFIER FOR WINE QUALITY (FULL PIPELINE)
# ------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
from tqdm import tqdm


# 20% test, 60% train, 20% validation
X_temp, X_test, y_temp, y_test = train_test_split(
    X_ros, y_ros, test_size=0.2, random_state=random_value, stratify=y_ros
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=random_value, stratify=y_temp
)
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# ------------------------------------------------------------
# 4. HYPERPARAMETER TUNING WITH K-FOLD CV + SMOTE
# ------------------------------------------------------------

# Decision Tree hyperparameter grid
max_depth_list   = [2, 4, 6, 8, 10, 20, None]
min_samples_split_list = [2, 5, 10, 20]
min_samples_leaf_list  = [1, 2, 4, 8]

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_value)

results = []
best_auc = -1
best_params = None

print("\nStarting hyperparameter search...\n")

for depth in max_depth_list:
    for min_split in min_samples_split_list:
        for min_leaf in tqdm(min_samples_leaf_list,
                             desc=f"depth={depth}, min_split={min_split}"):

            fold_aucs = []

            for train_idx, val_idx in kfold.split(X_train):

                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val   = X_train.iloc[val_idx]
                y_fold_val   = y_train.iloc[val_idx]

                # SMOTE only on training fold
                sm = SMOTE(random_state=random_value)
                X_res, y_res = sm.fit_resample(X_fold_train, y_fold_train)

                # Train Decision Tree
                dt = DecisionTreeClassifier(
                    max_depth=depth,
                    min_samples_split=min_split,
                    min_samples_leaf=min_leaf,
                    random_state=random_value
                )
                dt.fit(X_res, y_res)

                # Predict probabilities for ROC-AUC
                y_pred = dt.predict_proba(X_fold_val)[:, 1]
                auc = roc_auc_score(y_fold_val, y_pred)
                fold_aucs.append(auc)

            mean_auc = np.mean(fold_aucs)

            results.append((depth, min_split, min_leaf, mean_auc))

            print(f"depth={depth}, min_split={min_split}, "
                  f"min_leaf={min_leaf}, AUC={mean_auc:.4f}")

            # Track the best parameters
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = (depth, min_split, min_leaf)

print("\nBest Hyperparameters Found:")
print("Max Depth:", best_params[0])
print("Min Samples Split:", best_params[1])
print("Min Samples Leaf:", best_params[2])
print(f"Best CV AUC: {best_auc:.4f}")





# In[55]:


# ------------------------------------------------------------
# 5. FINAL TRAINING USING BEST PARAMETERS
# ------------------------------------------------------------

print("\nTraining final model with best hyperparameters...")

best_depth, best_min_split, best_min_leaf = best_params

sm = SMOTE(random_state=random_value)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

final_dt = DecisionTreeClassifier(
    max_depth=best_depth,
    min_samples_split=best_min_split,
    min_samples_leaf=best_min_leaf,
    random_state=random_value
)

final_dt.fit(X_train_res, y_train_res)

# ------------------------------------------------------------
# 6. EVALUATION ON VALIDATION + TEST SET
# ------------------------------------------------------------

def evaluate(model, X, y, name="Set"):
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

# Evaluate
evaluate(final_dt, X_val, y_val, "Validation")
evaluate(final_dt, X_test, y_test, "Test")


# # XGBoost Classifier

# In[57]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import xgboost as xgb

# 20% test, 60% train, 20% validation
X_temp, X_test, y_temp, y_test = train_test_split(    X_ros, y_ros, test_size=0.2, random_state=random_value, stratify=y_ros)
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

models = {
    "XGBoost": xgb.XGBClassifier(random_state=random_value, n_estimators=100, use_label_encoder=False, eval_metric='logloss')
}

best_auc = 0
best_depth = None
for depth in [2, 3, 5, 7]:
    model_xgb = xgb.XGBClassifier(random_state=random_value, max_depth=depth, n_estimators=100, use_label_encoder=False, eval_metric='logloss')
    model_xgb.fit(X_train_res, y_train_res)
    y_val_pred = model_xgb.predict(X_val)
    auc = roc_auc_score(y_val, y_val_pred)
    print(f"XGBoost max_depth={depth}, Validation ROC-AUC={auc:.3f}")
    if auc > best_auc:
        best_auc = auc
        best_depth = depth
print(f"Best XGBoost max_depth: {best_depth}, ROC-AUC={best_auc:.3f}")

final_xgmodel = xgb.XGBClassifier(random_state=random_value, max_depth=best_depth, n_estimators=100, use_label_encoder=False, eval_metric='logloss')
final_xgmodel.fit(X_train_res, y_train_res)
y_test_pred = final_xgmodel.predict(X_test)

print("\n--- Final XGBoost Test Evaluation ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_test_pred))




# In[58]:


# ------------------------------------------------------------
# 6. EVALUATION ON VALIDATION + TEST SET
# ------------------------------------------------------------

def evaluate(model, X, y, name="Set"):
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

# Evaluate
evaluate(final_xgmodel, X_val, y_val, "Validation")
evaluate(final_xgmodel, X_test, y_test, "Test")


# In[137]:





# In[ ]:




