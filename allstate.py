
from keras.models import Sequential
from keras import backend as K
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import sgd as SGD
from keras.layers.advanced_activations import ELU
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import train_test_split, KFold
import xgboost as xgb
from sklearn.preprocessing import normalize, RobustScaler

type=1

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_id = test.id
test = test.drop(["id"],axis=1)
X = train.drop(["loss","id"],axis=1)
y = train.loss.values
if type==1:
    X=pd.get_dummies(X, columns=X.columns[:116])
    test_enc=pd.get_dummies(test, columns=test.columns[:116])
    remove=[]
    for i in X.columns:
        if i not in test_enc.columns:
            remove.append(i)
    X.drop(remove, axis=1, inplace=True)

    remove=[]
    for i in test_enc.columns:
        if i not in X.columns:
            remove.append(i)
    test_enc.drop(remove, axis=1, inplace=True)
else:
    ntrain = train.shape[0]
    ntest = test.shape[0]
    features = train.columns

    cats = [feat for feat in features if 'cat' in feat]
    train_test = pd.concat((X, test)).reset_index(drop=True)
    for feat in cats:
        train_test[feat] = pd.factorize(train_test[feat], sort=True)[0]
    X = train_test.iloc[:ntrain, :]
    test_enc = train_test.iloc[ntrain:, :]
    X = X.fillna(method='ffill')
    test_enc = test_enc.fillna(method='ffill')

rsc=RobustScaler()
X=normalize(rsc.fit_transform(np.array(X)), axis=0,norm='max' )
test_norm=normalize(rsc.fit_transform(np.array(test_enc)), axis=0, norm='max')

def xfold(train, target, test):
    folds = 10
    models=0
    cv_sum = 0
    keraPred = []
    xgbPred=[]
    kf = KFold(train.shape[0], n_folds=folds)
    for i, (train_index, test_index) in enumerate(kf):
        print('\n Fold %d\n' % (i + 1))
        X_train, X_val = train[train_index], train[test_index]
        y_train, y_val = target[train_index], target[test_index]

        ##Model
        model = Sequential()
        act = ELU(input_shape=(X_train.shape[1],))
        model.add(Dropout(0.2, input_shape=(X_train.shape[1],)))
        model.add(Dense(386, input_shape=(X_train.shape[1],),init='he_normal'))
        model.add(act)
        model.add(Dropout(0.2))
        model.add(Dense(256, init='he_normal', activation='relu'))
        model.add(act)
        model.add(Dropout(0.2))
        model.add(Dense(128, init='normal', activation='relu'))
        model.add(Dense(1, init='normal'))
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9)


        model.compile(optimizer='adam', loss='mae')
        modelsaver=ModelCheckpoint(filepath="weights.hdf5", verbose=0, save_best_only=True)
        #model.fit(X_train, y_train, nb_epoch=1, validation_data=(X_val, y_val),callbacks=[modelsaver])

        model.load_weights("weights.hdf5")
        print("Loaded model from disk")
        scores_val = model.predict(X_val)
        cv_score = mean_absolute_error(y_val, scores_val)
        print(' eval-MAE keras: %.6f' % cv_score)
        y_pred_inner = model.predict(test)

        ####################################
        #  Add Predictions and Average Them
        ####################################

        keraPred.append(y_pred_inner)
        if models>0:
            margPred += y_pred_inner
            models+=1
        else:
            margPred=y_pred_inner
            models+=1
        cv_sum = cv_sum + cv_score

        # apply xgboost
        d_train = xgb.DMatrix(X_train, np.log(y_train))
        d_valid = xgb.DMatrix(X_val, np.log(y_val))
        d_test = xgb.DMatrix(test)

        xgb_params = {
            'seed': 120,
            'colsample_bytree': 0.7,
            'silent': 1,
            'subsample': 0.7,
            'learning_rate': 0.02,
            'objective': 'reg:linear',
            'max_depth': 7,
            'min_child_weight': 1,
        }
        params = {}
        params['booster'] = 'gbtree'
        params['objective'] = "reg:linear"
        params['eta'] = 0.1
        params['gamma'] = 0.5290
        params['min_child_weight'] = 4.2922
        params['colsample_bytree'] = 0.3085
        params['subsample'] = 0.9930
        params['learning_rate']=0.01
        params['max_depth'] = 7
        params['max_delta_step'] = 0
        params['silent'] = 1
        params['seed'] = 120


        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        def xg_eval_mae(yhat, dtrain):
            y = dtrain.get_label()
            return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))

        clf = xgb.train(params, d_train,250000, watchlist, early_stopping_rounds=30,
                        verbose_eval=20, maximize=False, feval=xg_eval_mae)

        scores_val = np.exp(clf.predict(d_valid))
        cv_score = mean_absolute_error(y_val, scores_val)
        print(' eval-MAE xgb: %.6f' % cv_score)
        clfPred=np.exp(clf.predict(d_test))
        xgbPred.append(clfPred)
        clfPred=np.vstack(clfPred)
        cv_sum = cv_sum + cv_score
        if models > 0:
            margPred += clfPred
            models+=1
        else:
            margPred = clfPred
            models+=1


    mpred = margPred / models
    score = cv_sum / models
    print('\n Average eval-MAE: %.6f' % score)

    return mpred


y_pred=xfold(X,y,test_norm)
submission = pd.DataFrame({"id":test_id, "loss":y_pred[:,0]})
submission.to_csv("submission2.csv", index=False)