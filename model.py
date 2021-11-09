import numpy as np
import pandas as pd
import os

import sklearn
#import sklearn.grid_search
import sklearn.calibration
import sklearn.neighbors
import sklearn.discriminant_analysis
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from skopt import BayesSearchCV
from skopt.space import Real

from matplotlib import pyplot as plt


#Load image dataset
FEATURE_PATH = './data/HBS_cropped_images/cnn_feature/norm_images'
category = 'HER2'

# read excel sheet
df = pd.read_csv('HBS_final_dataset_old.csv')
X = []
Y = []

exclude_list = ['PATIENT_ID', 'ER', 'PR', 'p53', 'HER2', 'PT_SBST_NO', 'EstimateImmuneScore']
cols = [col for col in df.columns if col not in exclude_list]
df_sub = df[cols[1:]][df_features]
print(df_sub)

samples = os.listdir(FEATURE_PATH)
for sample in df['PATIENT_ID'].tolist():
    if not sample in os.listdir(FEATURE_PATH):
        continue
    
    #Read features as X data
    fn_path = os.path.join(FEATURE_PATH, sample)
    fn_num = 0
    feats = []
    for fn in os.listdir(fn_path):
        fn_num += 1
        feat = np.load(os.path.join(fn_path, fn)).reshape(-1)# read feature maps from histology images
        feat = np.append(feat, df_sub.iloc[df.index[df['PATIENT_ID'] == sample].tolist()].values) #append clinical data to feature maps
        
        feats.append( feat )#read features collapsed into 1D
    
    X.append(feats)
    
    print('%s %d'%(sample, fn_num))
    Y.extend(df[category].iloc[df.index[df['PATIENT_ID'] == sample].tolist()].to_numpy())
    #Read corresponding Y labelprint(df.index[df['PATIENT_ID'] == sample])
#     if df[category].iloc[df.index[df['PATIENT_ID'] == sample].tolist()[0]] != 'N':
#         Y.extend([1])        
#     else:
#         Y.extend([0])
Y = np.array(Y)


# build train/test sets
idx = np.arange(len(X))
cv_folds = 5
skf = StratifiedKFold( n_splits=cv_folds, shuffle=True )
idx_train_test = list(skf.split(idx, Y))


#linear SVM training
options = {}
options['kernel'] = 'linear'
options['classifier'] = 'svm'
sw = None
print(category)


#prediction and metric calculation
res = {"acc":[],"f_score":[],"confusion":[], 'auc':[], 'roc':[], 'prc':[], 'auprc':[], 'precision':[], 'recall':[]}
pred_level = 'instance'

for train_index, test_index in idx_train_test:
    

    X_train = []
    y_train = []
    for idx in train_index:
        #X_train.append(np.squeeze(X[idx]))
        X_train.append(np.vstack(X[idx]))
        y_train.extend([Y[idx]]*len(X[idx]))
        #[y_train.append(Y[idx]) for _ in range(len(X[idx]))]


    X_test = []
    y_test_inst = []
    y_test = Y[test_index]#y_test takes bag-level labels
    tile_no_test = []
    for idx in  test_index:
        X_test.append(np.vstack(X[idx]))
        y_test_inst.extend([Y[idx]]*len(X[idx]))
        tile_no_test.append(len(X[idx]))

    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)

    y_train = np.array(y_train)
    y_test_inst = np.array(y_test_inst)
    

    #calculate sample weight
    sw = compute_sample_weight(class_weight='balanced', y=y_train)
    model = LinearClassifier( n_jobs=7, **options )
    model.fit( X_train, y_train, calibrate=True, param_search=True, sample_weight=sw )

    if pred_level == 'bag':

        prev_idx = 0
        y_p = []

        flag = -3
        y_p_hist = []
        for i, idx in enumerate(tile_no_test):
            y_p_inst = (model.predict(X_test[prev_idx:prev_idx+idx]))
            y_p.append(np.mean(y_p_inst, axis=0))
            y_p_hist.append(np.array(y_p_inst[:, 1]))
            prev_idx = idx

        y_p = np.squeeze(y_p)

        #Store metrics for each fold
        y_predict = np.argmax(np.array(y_p), axis=1)
        acc = sklearn.metrics.accuracy_score( y_test, y_predict )
        f_score = sklearn.metrics.f1_score( y_test, y_predict )

        confusion = sklearn.metrics.confusion_matrix( y_test, y_predict )

        y_p = y_p[:,1]

        fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_p)
        precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, y_p )


    elif pred_level == 'instance':   
        y_p = model.predict(X_test)

        #Store metrics for each fold
        y_predict = np.argmax(y_p, axis=1)
        acc = sklearn.metrics.accuracy_score( y_test_inst, y_predict )
        f_score = sklearn.metrics.f1_score( y_test_inst, y_predict )

        confusion = sklearn.metrics.confusion_matrix( y_test_inst, y_predict )

        y_p = y_p[:,1]

        fpr, tpr, _ = sklearn.metrics.roc_curve(y_test_inst, y_p)
        precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test_inst, y_p)


    res['acc'].append(acc)
    res['f_score'].append(f_score)
    res['confusion'].append(confusion)
    res['auc'].append(sklearn.metrics.auc(fpr, tpr))
    res['roc'].append([fpr, tpr])
    res['prc'].append([precision, recall])
    res['auprc'].append(sklearn.metrics.auc(recall, precision))
    print(res['acc'], res['f_score'], res['auc'])
    res['precision'].append(precision)
    res['recall'].append(recall)

    if len(res['acc']) < 5:
        del model

