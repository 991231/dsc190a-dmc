import regex
import string
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer

##### column types #####
text_cols_removed = ["name", "host_name", "summary", "host_about"]
text_cols_binarize = [
    "interaction",
    "house_rules",
    "notes",
    "space",
    "neighborhood_overview",
]
text_cols_keywords = ["transit", "access"]
text_cols_find_CJK = ["description"]
bool_cols = [
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "instant_bookable",
    "is_business_travel_ready",
    "require_guest_profile_picture",
    "require_guest_phone_verification",
]
id_cols = ["id", "host_id"]
datetime_cols = ["host_since", "first_review", "last_review"]
category_cols = ["property_type", "room_type", "bed_type"]
ordinal_cols = [
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "host_response_time",
    "cancellation_policy",
]
cat_location_cols_removed = [
    "city",
    "state",
    "zipcode",
    "market",
    "country_code",
    "country",
    "host_location",
]
# cat_location_cols_keep =
category_cols += [
    "host_neighbourhood",
    "neighbourhood_cleansed",
    "neighbourhood_group_cleansed",
]
cat_multi_cols = ["host_verifications", "amenities"]
numeric_cols = [
    "host_listings_count",
    "calculated_host_listings_count",
    "minimum_nights",
    "maximum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "accommodates",
    "guests_included",
    "extra_people",
    "review_scores_rating",
    "host_response_rate",
    "beds",
    "bathrooms",
    "bedrooms",
]
numeric_cols_removed = ["square_feet"]
all_nan_cols = ["experiences_offered", "host_acceptance_rate"]

##### feature engineering #####
def feature_engineering(X):

    groupby_col = "neighbourhood_cleansed"

    X = X.copy()
    to_drop = (
        all_nan_cols
        + text_cols_removed
        + cat_location_cols_removed
        + numeric_cols_removed
    )
    X = X.drop(to_drop, axis=1)

    # ordinal, manual assign order
    X["host_response_time"] = X["host_response_time"].replace(
        {
            item: idx
            for idx, item in enumerate(
                [
                    "within an hour",
                    "within a few hours",
                    "within a day",
                    "a few days or more",
                ]
            )
        }
    )

    X["cancellation_policy"] = X["cancellation_policy"].replace(
        {
            item: idx
            for idx, item in enumerate(
                [
                    "long_term",
                    "super_strict_60",
                    "super_strict_30",
                    "strict_14_with_grace_period",
                    "strict",
                    "moderate",
                    "flexible",
                ]
            )
        }
    )

    # clip outliers
    X["maximum_nights"] = X["maximum_nights"].clip(0, 400)

    # new numerical features
    X["months_opened"] = (X["number_of_reviews"] / X["reviews_per_month"]).fillna(0)
    X["reviews_per_month"] = X["reviews_per_month"].fillna(0)

    # strip symbols
    X["host_response_rate"] = X["host_response_rate"].apply(
        lambda x: float(x[:-1]) / 100 if not pd.isnull(x) else x  # strip %
    )
    X["extra_people"] = X["extra_people"].apply(
        lambda x: float(x[1:]) if not pd.isnull(x) else x  # strip $
    )

    # id, take log (maybe not useful)
    for col in id_cols:
        X[col] = np.log1p(X[col])

    # boolean, binarize
    for col in bool_cols:
        X[col] = X[col].replace({"t": 1, "f": 0, np.NaN: 0})

    # datetime, impute mean, shift to make min 0, convert to day
    for col in datetime_cols:
        temp = pd.to_datetime(X[col])
        temp = temp.fillna(temp.mean())
        temp = (temp - temp.min()).dt.days
        X[col] = temp
        numeric_cols.append(col)

    # categorical, impute sample
    for col in category_cols:
        temp = X[col]
        sample = (
            temp[~temp.isnull()]
            .sample(n=len(temp), replace=True)
            .reset_index(drop=True)
        )
        X[col] = temp.fillna(sample)

    # ordinal, impute sample by group
    for col in ordinal_cols:
        temp = X[[groupby_col, col]]
        imputed = temp[col].fillna(
            temp.groupby(groupby_col)[col].transform(sample_imputer)
        )
        X[col] = imputed.fillna(sample_imputer(imputed)).astype(int)

    # numerical, impute mean by group
    for col in numeric_cols:
        temp = X[[groupby_col, col]]
        X[col] = X[col].fillna(temp.groupby(groupby_col)[col].transform("mean"))
        X[col] = X[col].fillna(X[col].mean())

    return X


def sample_imputer(g):
    g_nonnull = g[~g.isnull()]
    if len(g_nonnull) == 0:
        return np.NaN
    return g_nonnull.sample(n=len(g), replace=True).reset_index(drop=True)


def clean_text(s):
    if pd.isnull(s):
        return s
    s = regex.sub(f"[{string.punctuation}]", " ", s)
    s = regex.sub("\s+", " ", s)
    s = s.lower().strip()
    return s


matcher = lambda pat, s: (not pd.isnull(s)) and (regex.search(pat, s) is not None)


def extract_keyword(col, keywords):
    onehot = (
        lambda col, pat: col.apply(lambda s: matcher(pat, s))
        .astype(int)
        .rename(col.name + "_" + pat)
    )

    out_cols = []
    [out_cols.append(onehot(col, k)) for k in keywords]
    return pd.concat(out_cols, axis=1)


# hardcoded from text feature EDA
keywords_vocab = {
    "transit": [
        "subway|train",
        "bus",
        "taxi|uber|lyft|zipcar",
        "parking|lot",
        "airport",
        "ferry",
        "citi",
        "express",
        "walk",
    ],
    "access": [
        "kitchen",
        "bathroom",
        "apartment|apt",
        "shared",
        "common",
        "bedroom",
        "microwave",
        "gym",
        "fridge",
        "refrigerator",
        "food",
        "garden",
        "closet",
        "shower",
        "cooking",
        "backyard",
        "toaster",
        "terrace",
        "oven",
    ],
}


def feature_engineering_combined(X_combined):

    # multi-value features, binarize
    mv_features = []
    for col in cat_multi_cols:
        mv_features.append(MultiLabelBinarizer().fit_transform(X_combined.pop(col)))

    # text, extract keywords from hardcoded dictionary
    keyword_features = []
    for col in text_cols_keywords:
        keyword_features.append(
            extract_keyword(
                X_combined[col].apply(clean_text), keywords_vocab[col]
            ).values
        )

    # text, binarize (has contents or not)
    for col in text_cols_binarize:
        X_combined[col] = (
            X_combined[col].apply(lambda s: 0 if pd.isnull(s) else 1).astype(int)
        )

    # find Chinese, Jacpanese, Korean houses
    for col in text_cols_find_CJK:  # only has 1 column ("description") now
        temp = X_combined.pop(col)
        is_CJK = extract_keyword(temp, ["[\p{Hiragana}\p{Katakana}]+", "[\p{Hangul}]+"])
        is_cn = temp.apply(
            lambda s: not matcher("[\p{Hiragana}\p{Katakana}]+", s)
            and matcher("[\p{Han}]+", s)
        ).astype(int)
        is_CJK = pd.concat([is_CJK, is_cn], axis=1).values

    # drop all raw text columns
    for col in set(text_cols_keywords) - set(text_cols_binarize):
        X_combined.pop(col)

    # categorical, one-hot
    ct = ColumnTransformer(
        [("onehot", OneHotEncoder(), category_cols)], remainder="passthrough"
    )
    X_processed = ct.fit_transform(X_combined).toarray()

    # combine
    X_processed = np.hstack([X_processed] + mv_features + keyword_features + [is_CJK])
    return X_processed
