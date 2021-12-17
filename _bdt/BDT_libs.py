import xgboost as xgb


def Run(data, training, objective, predict, features, n=1000, param=None):
    if param == None:
        param = {}

        # Booster parameters
        param['eta']              = 0.3 # learning rate
        param['gamma']            = 0.5
        param['max_depth']        = 5 # maximum depth of a tree
        param['subsample']        = 1 # fraction of events to train tree on
        param['colsample_bytree'] = 0.2 # fraction of features to train tree on

        # Learning task parameters
        param['objective']   = objective # objective function
        param['eval_metric'] = 'error'           # evaluation metric for cross validation
        param = list(param.items()) + [('eval_metric', 'logloss')] + [('eval_metric', 'rmse')]

    # Assign variable to predict and features
    target_train = training[predict]
    train_DM = xgb.DMatrix(training[features], target_train)
    
    true = data[predict]
    test_DM = xgb.DMatrix(data[features], true)

    # create watchlist to track evaluation metrics per iteration
    watchlist = [(test_DM,'test'), (train_DM,'train')]
    results = {}

    model = xgb.train(param, train_DM, num_boost_round=n, evals=watchlist, evals_result=results, verbose_eval=False) # train model
    prediction = model.predict(test_DM) # use model on data

    return prediction, true, target_train, results, model
