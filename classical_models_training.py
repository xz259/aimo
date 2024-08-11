import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Ridge
from xgboost import XGBClassifier
from sklearn.svm import SVC
import pickle

from scipy.sparse import load_npz
X_train = load_npz('/data/processed_train.npz')

train = pd.read_csv('/data/train_essays.csv')
Y_train = train['label'].values

mnb = MultinomialNB(alpha=1e-5)
mnb.fit(X_train, Y_train)

output_path = '/model_checkpoints/mnb_model.pkl'

with open(output_path, 'wb') as file:
    pickle.dump(mnb, file)

ridge = Ridge(solver='sag', max_iter=5000, tol=1e-4, alpha=1)
ridge.fit(X_train, Y_train)

output_path = '/model_checkpoints/ridge_model.pkl'

with open(output_path, 'wb') as file:
    pickle.dump(ridge, file)

xgb = XGBClassifier(n_estimators=500,
                    learning_rate=0.1,
                    subsample=0.6,
                    colsample_bytree=0.7,
                    colsample_bylevel=0.3,
                    use_label_encoder=False,
                    reg_alpha=0.5,
                    reg_lambda=0.5,
    )

xgb.fit(X_train, Y_train)

output_path = '/model_checkpoints/xgb_model.pkl'

with open(output_path, 'wb') as file:
    pickle.dump(xgb, file)



svc= SVC(
        tol=1e-4, 
        C=1e-5,           
        kernel='linear',    
        gamma='auto',
        probability=True
    )

svc.fit(X_train, Y_train)

output_path = '/model_checkpoints/svc_model.pkl'

with open(output_path, 'wb') as file:
    pickle.dump(svc, file)
    
