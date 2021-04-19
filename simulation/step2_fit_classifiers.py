import os
import pickle
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from tqdm import tqdm


def get_metric(y, yp):
    ## assumes binary class
    auc = roc_auc_score(y, yp[:,1])
    
    fpr, tpr, tt = roc_curve(y, yp[:,1])
    best_thres = tt[np.argmin(fpr**2+(1-tpr)**2)]
    yp2 = (yp[:,1]>best_thres).astype(int)
    acc = accuracy_score(y, yp2)
    bacc = balanced_accuracy_score(y, yp2)
    kappa = cohen_kappa_score(y, yp2)
    f1 = f1_score(y, yp2)
    
    res = pd.DataFrame(data={'metric':np.r_[
            auc, acc, bacc, kappa, f1 
            ]},
            index=[
                'AUC', 'Accuracy', 'BalancedAccuracy', 'CohenKappa', 'F1Score',
            ])
    return res
    
    
def fit_model(X, y, Ncv, model_type='logreg', best_params=None, n_jobs=1, random_state=None):
    cv_scores_tr = []
    cv_scores_te = []
    models = []
    Xtes = []
    ytes = []
    yptes = []
    if best_params is None:
        params = []
    else:
        params = best_params
        
    cv = StratifiedKFold(n_splits=Ncv, shuffle=True, random_state=random_state)
    for cvi, (trid, teid) in enumerate(cv.split(X, y)):
        Xtr = X[trid]
        ytr = y[trid]
        
        # normalize
        ss = StandardScaler().fit(Xtr)
        Xtr = ss.transform(Xtr)
    
        if model_type=='logreg':
            model = LogisticRegression(
                penalty='elasticnet', class_weight='balanced',
                random_state=random_state, solver='saga',
                max_iter=1000, n_jobs=n_jobs)
            model_params  = {'C':[0.1,1,10], 'l1_ratio':[0.5,0.6,0.7,0.8,0.9]}
        elif model_type=='rf':
            model = RandomForestClassifier(
                n_jobs=n_jobs, min_samples_leaf=10,
                random_state=random_state, class_weight='balanced')
            model_params = {'n_estimators':[10,50,100], 'max_depth':[3,5], 'ccp_alpha':[0.1,1,10]}
        else:
            raise ValueError('Unknown model_type: %s'%model_type)
            
        if best_params is None:
            if len(teid)>0:  # in fold
                if hasattr(model, 'n_jobs'):
                    model.n_jobs = 1
                if len(model_params)>0:
                    model = GridSearchCV(model, model_params,
                            n_jobs=n_jobs, refit=True,
                            cv=Ncv,
                            verbose=False)
            elif len(params)>0:  # refit
                for p in params[0]:
                    val = Counter([param[p] for param in params]).most_common()[0][0]
                    if '__' in p:
                        pp = p.split('__')
                        exec('model.%s.%s = %f'%(pp[0], pp[1], val))  # TODO assumes two
                    else:
                        exec('model.%s = %f'%(p, val))
                
        elif len(params)>0:
            for p in params[cvi]:
                if '__' in p:
                    pp = p.split('__')
                    exec('model.%s.%s = %f'%(pp[0], pp[1], params[cvi][p]))  # TODO assumes two
                else:
                    exec('model.%s = %f'%(p, params[cvi][p]))
            if hasattr(model, 'n_jobs'):
                model.n_jobs = n_jobs
        model.fit(Xtr, ytr)
        if best_params is None and hasattr(model, 'best_params_'):
            params.append({p:model.best_params_[p] for p in model_params})
        if hasattr(model, 'best_estimator_'):
            model = model.best_estimator_
        
        calibrated_model = CalibratedClassifierCV(base_estimator=model, cv='prefit')
        calibrated_model.fit(Xtr, ytr)
        
        yptr = calibrated_model.predict_proba(Xtr)
        
        model = Pipeline([
            ('standardizer', ss),
            ('predictor', calibrated_model),
            ])
        models.append(model)
        
        if len(teid)>0:
            Xte = X[teid]
            yte = y[teid]
            ypte = model.predict_proba(Xte)
            
            cv_scores_tr.append(get_metric(ytr, yptr))
            cv_scores_te.append(get_metric(yte, ypte))
            ytes.append(yte)
            yptes.append(ypte)
            Xtes.append(Xte)
            
    cv_scores_tr = sum(cv_scores_tr)/len(cv_scores_tr)
    cv_scores_te = sum(cv_scores_te)/len(cv_scores_te)
    
    return cv_scores_tr, cv_scores_te, params, models, Xtes, ytes, yptes, yptr
    
    
if __name__=='__main__':
    simulation_data_dir = 'datasets'
    df = pd.read_csv(os.path.join(simulation_data_dir, 'simulator_classification_dataset_list.csv'))
    Ns = [100,1000,10000]
    model_types = ['logreg', 'rf']
    random_state = 2021
    n_jobs = 8
    Ncv = 5
    output_dir = 'step2_output_models_predictions'
    
    import pdb;pdb.set_trace()
    for i in tqdm(range(len(df))):
        for n in Ns:
            # generate different training sizes
            df_sim = pd.read_csv(df.Path.iloc[i])
            df_sim = df_sim[:n]  # because datset is shuffled, taking first N approximately keeps the class ratio
            #TODO assumes event is the last column
            X = df_sim.values.astype(float)[:,:-1]
            y = df_sim.values[:,-1].astype(int)
            
            for model_type in model_types:
                cv_scores_tr, cv_scores_te, params, models, Xtes, ytes, yptes, yp = fit_model(X, y, Ncv, model_type=model_type, n_jobs=n_jobs, random_state=random_state)
                model = models[-1]
                
                # save model
                save_file_name = os.path.basename(df.Path.iloc[i]).replace('.csv',f'_model_N{n}_model{model_type}.pickle')
                with open(os.path.join(output_dir, save_file_name), 'wb') as ff:
                    pickle.dump(model, ff)
                    
                # save prediction
                save_file_name = os.path.basename(df.Path.iloc[i]).replace('.csv',f'_prediction_N{n}_model{model_type}.csv')
                df_yp = pd.DataFrame(data={'p':yp})
                df_yp.to_csv(os.path.join(output_dir, save_file_name), index=False)
    
