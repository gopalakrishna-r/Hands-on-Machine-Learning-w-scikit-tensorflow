import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import shift
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

from util.selector.Selector import TopFeatureSelector


def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


from zlib import crc32


# determine whether the test data instance's identifier falls within the ratio
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) ^ 0xfffffff < test_ratio * 2 ** 32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    print(in_test_set)
    return data.iloc[~in_test_set], data.iloc[in_test_set]


def display_scores(scores):
    print("scores:", scores)
    print("mean:", scores.mean())
    print("standard deviation", scores.std())


def build_transformer(housing_numericals, cat_attribs=None):
    if cat_attribs is None:
        cat_attribs = ["ocean_proximity"]
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from util.transformer import CombinedAttributesAdder
    num_attribs = list(housing_numericals)

    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    return ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', OneHotEncoder(), cat_attribs)
    ])


def build_transformer_with_features_model(housing_numericals, feature_count, top_features=None, cat_attribs=None,
                                          best_parameters=None):
    pipeline = build_transformer(housing_numericals, cat_attribs)
    if top_features is not None:
        return Pipeline([
                ('preparation', pipeline),
                ('feature_selection', TopFeatureSelector(top_features, feature_count))
            ]) if best_parameters is None else Pipeline([
                ('preparation', pipeline),
                ('feature_selection', TopFeatureSelector(top_features, feature_count)),
                ('svm_reg', SVR(**best_parameters))
            ])
    if best_parameters is None:
        return Pipeline([
            ('preparation', pipeline)
        ])
    else:
        return Pipeline([
            ('preparation', pipeline),
            ('svm_reg', SVR(best_parameters))
        ])


def predict_with_best_model(X_test, y_test, full_pipeline, final_model):
    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(f"final prediction score {final_rmse}")
    # checking the precision of the model using confidence level
    from scipy import stats
    confidence = 0.95
    squared_errors = (final_predictions - y_test) ** 2
    return np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                                                loc=squared_errors.mean(),
                                                scale=stats.sem(squared_errors)))


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


def cross_val_score_wth_split(X_train, y_train, model):
    skfolds = StratifiedKFold(n_splits=3, random_state=42)
    for train_index, test_index in skfolds.split(X_train, y_train):
        clone_clf = clone(model)
        X_train_folds = X_train[train_index]
        y_train_folds = y_train[train_index]
        X_test_folds = X_train[test_index]
        y_test_folds = y_train[test_index]

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone.predit(X_test_folds)

        n_correct = sum(y_pred == y_test_folds)
        print(n_correct / len(y_pred))


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim(0, -1)
    plt.xlim([-700000, 700000])
    plt.show()


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=mpl.cm.binary, **options)
    plt.axis("off")


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    plt.show()


def shift_image(image, x, y):
    im = image.reshape((28, 28))
    shifted_image = shift(im, [x, y], cval=0, mode='constant')
    return shifted_image.reshape([-1])

# plot learning curve
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curve(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth = 2, label = "train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth = 3, label = "val")
    plt.show()
