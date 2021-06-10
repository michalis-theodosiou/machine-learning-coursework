import numpy as np
import random


def split_forest(x, y, max_depth, feature_frac, min_leaf_size, depth=0):

    if depth == 0:
        global col_dict
        global feature_split
        col_dict = {x[0]: x[1] for x in zip(range(len(x.columns)), x.columns.tolist())}
        feature_split = x.shape[1] // feature_frac

    if depth == max_depth or len(y) <= min_leaf_size:
        return {'prediction': np.mean(y)}

    x_cols_sample = random.sample(list(col_dict), feature_split)
    rule = find_split_forest(x.to_numpy()[:, x_cols_sample], y, x_cols_sample)

    if rule['feature'] is None:
        return {'prediction': np.mean(y)}

    split_ix = x[rule['feature']].to_numpy() < rule['threshold']

    rule['left'] = split_forest(x[split_ix], y[split_ix], max_depth,
                                feature_frac, min_leaf_size, depth=depth+1)
    rule['right'] = split_forest(x[~split_ix], y[~split_ix], max_depth,
                                 feature_frac, min_leaf_size, depth=depth+1)

    return rule


def split_forest_f(x, y, max_depth, min_leaf_size, depth=0):

    if depth == 0:
        global col_dict
        col_dict = {x[0]: x[1] for x in zip(range(len(x.columns)), x.columns.tolist())}

    if depth == max_depth or len(y) <= min_leaf_size:
        return {'prediction': np.mean(y)}

    x_cols_sample = list(col_dict)
    rule = find_split_forest(x.to_numpy(), y, x_cols_sample)

    if rule['feature'] is None:
        return {'prediction': np.mean(y)}

    split_ix = x[rule['feature']].to_numpy() < rule['threshold']

    rule['left'] = split_forest_f(x[split_ix], y[split_ix], max_depth,
                                  min_leaf_size, depth=depth+1)
    rule['right'] = split_forest_f(x[~split_ix], y[~split_ix], max_depth,
                                   min_leaf_size, depth=depth+1)

    return rule


def find_split_forest(x, y, x_cols_sample):
    feature_b, threshold_b, min_error = None, None, np.inf

    for ix, column in enumerate(x.T):
        thresholds = np.unique(column)[1:]
        for t in thresholds:
            y_l_ix = x[:, ix] < t
            y_l, y_r = y[y_l_ix], y[~y_l_ix]
            error = find_error(y_l, y_r)
            if error < min_error:
                min_error = error
                threshold_b = t
                feature_b = col_dict[x_cols_sample[ix]]

    return {'feature': feature_b, 'threshold': threshold_b}


def find_error(y_l, y_r):
    rss_l = (np.square(y_l - y_l.mean())).sum()
    rss_r = (np.square(y_r - y_r.mean())).sum()
    return rss_l + rss_r
