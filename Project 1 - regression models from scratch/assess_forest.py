import numpy as np


def predict(point, rules):
    prediction = None
    while prediction is None:
        feature, threshold = rules['feature'], rules['threshold']
        if point[feature] < threshold:
            rules = rules['left']
        else:
            rules = rules['right']
        prediction = rules.get('prediction', None)
    return prediction


def pred_x(x, rule):
    return x.apply(lambda x: predict(x, rule), axis=1)


def calc_metrics(y_exp, y_pred):

    mae = np.mean(abs(y_exp - y_pred))
    mse = np.mean(np.square(y_exp - y_pred))
    rmse = np.sqrt(mse)

    sey = sum(np.square(y_exp-y_exp.mean()))
    ser = sum(np.square(y_exp-y_pred))
    r2 = 1 - (ser/sey)

    return mse, rmse, mae, r2


def rmses(y_preds, y_exp):
    y_results = []
    for prediction in y_preds:
        _, y_res, _, _ = calc_metrics(y_exp, prediction)
        y_results.append(y_res)
    return y_results


def maes(y_preds, y_exp):
    y_results = []
    for prediction in y_preds:
        _, _, y_res, _ = calc_metrics(y_exp, prediction)
        y_results.append(y_res)
    return y_results
