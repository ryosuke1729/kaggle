import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
import nltk


import re
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from geopy.geocoders import Nominatim
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

path = "/content/drive/MyDrive/Colab Notebooks/Kaggle/minpaku/"

df_train = pd.read_csv('/content/train.csv')
df_test = pd.read_csv('/content/test.csv')

df_test['y'] = np.nan
df = pd.concat([df_train, df_test], ignore_index=True, sort=False)

# NaN
nan_columns = [col for col in df.columns if df[col].isna().any()]
for col in nan_columns:
    new_col_name = f"{col}_valid"
    df[new_col_name] = df[col].notna().astype(int)

#ame
dfa = df['amenities'].apply(lambda x: x.strip('{}').replace('"', '').split(','))
amenities_df = dfa.str.join('|').str.get_dummies()
df = pd.concat([df, amenities_df], axis=1)
df.drop(columns=['amenities'], inplace=True)
df["amenities_number"] = amenities_df.sum(axis=1)

#dummy
columns_to_dummy = ["bed_type", "cancellation_policy", "city", "cleaning_fee", "instant_bookable", "property_type", "room_type"]
dummy_df = pd.get_dummies(df[columns_to_dummy], drop_first=True)
df = pd.concat([df, dummy_df], axis=1)
df.drop(columns=columns_to_dummy, inplace=True)

#cleanstr
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text
df['cleaned_name'] = df['name'].apply(preprocess_text)
df['cleaned_description'] = df['description'].apply(preprocess_text)
def count_words(text):
    words = re.findall(r'\b\w+\b', text)
    return len(words)
df['name_words'] = df['cleaned_name'].apply(count_words)
df['description_words'] = df['cleaned_description'].apply(count_words)
nltk.download('stopwords')
vectorizer = TfidfVectorizer(max_features=1000)

X = vectorizer.fit_transform(df['cleaned_name'])
X_train, X_test, y_train, y_test = train_test_split(X, df['y'], test_size=0.2, random_state=42)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt'
}
train_data = lgb.Dataset(X_train.astype(np.float32), label=y_train.astype(np.float32))
model = lgb.train(params, train_data, num_boost_round=100)
feature_importance = model.feature_importance()
feature_names = vectorizer.get_feature_names_out()
feature_importance_dict = dict(zip(feature_names, feature_importance))
top_features_n = sorted(feature_importance_dict, key=feature_importance_dict.get, reverse=True)[:50]

X = vectorizer.fit_transform(df['cleaned_description'])
X_train, X_test, y_train, y_test = train_test_split(X, df['y'], test_size=0.2, random_state=42)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt'
}
train_data = lgb.Dataset(X_train.astype(np.float32), label=y_train.astype(np.float32))
model = lgb.train(params, train_data, num_boost_round=100)
feature_importance = model.feature_importance()
feature_names = vectorizer.get_feature_names_out()
feature_importance_dict = dict(zip(feature_names, feature_importance))
top_features_d = sorted(feature_importance_dict, key=feature_importance_dict.get, reverse=True)[:50]

for feature in top_features_n:
    df[f'{feature}_count_name'] = df['cleaned_name'].apply(lambda x: x.count(feature))
for feature in top_features_d:
    df[f'{feature}_count_description'] = df['cleaned_description'].apply(lambda x: x.count(feature))

#neibo
dfk = df[{'latitude', 'longitude','neighbourhood'}]
df_missing = dfk[dfk['neighbourhood'].isnull()]
df_not_missing = dfk[dfk['neighbourhood'].notnull()]
X_train = df_not_missing[{'latitude', 'longitude'}]
Y_train = df_not_missing[{'neighbourhood'}]
X_test = df_missing[{'latitude', 'longitude'}]
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
df.loc[df['neighbourhood'].isna(), 'neighbourhood'] = Y_pred

#datatime
df['first_review'] = pd.to_datetime(df['first_review'])
df['last_review'] = pd.to_datetime(df['last_review'])
df['host_since'] = pd.to_datetime(df['host_since'])
df['hostfirst_between']  = (df['first_review'] - df['host_since']).dt.days
df['hostlast_between']  = (df['last_review'] - df['host_since']).dt.days
df['lastfirst_between']  = (df['last_review'] - df['first_review']).dt.days
df['first_review_year'] = df['first_review'].dt.year
df['first_review_month'] = df['first_review'].dt.month
df['first_review_day'] = df['first_review'].dt.day
df['last_review_year'] = df['last_review'].dt.year
df['last_review_month'] = df['last_review'].dt.month
df['last_review_day'] = df['last_review'].dt.day
df['host_since_year'] = df['host_since'].dt.year
df['host_since_month'] = df['host_since'].dt.month
df['host_since_day'] = df['host_since'].dt.day

#moromoro
df['host_response_rate'] = df['host_response_rate'].apply(lambda x: re.sub(r'\D', '', str(x)) if pd.notnull(x) else x)
df['host_response_rate'] = pd.to_numeric(df['host_response_rate'], errors='coerce', downcast='integer')
df['host_has_profile_pic'] = df['host_has_profile_pic'].replace({'t': 1, 'f': 0})
df['host_identity_verified'] = df['host_identity_verified'].replace({'t': 1, 'f': 0})

#zipcode
df["zipcode"][133] = np.nan
df["zipcode"][7692] = np.nan
df["zipcode"][11983] = np.nan
df["zipcode"][16909] = np.nan
def get_zipcode(latitude, longitude):
    if pd.isna(latitude) or pd.isna(longitude):
        return None
    geolocator = Nominatim(user_agent="geo_locator")
    location = geolocator.reverse((latitude, longitude), language="en")
    address = location.raw.get("address", {})
    zipcode = address.get("postcode", None)
    return str(zipcode)
tqdm.pandas(desc="Geocoding Progress")
df.loc[df['zipcode'].isnull(), 'zipcode'] = df.loc[df['zipcode'].isnull()].progress_apply(lambda row: get_zipcode(row["latitude"], row["longitude"]), axis=1)
df["zipcode"][133] = "90265"
df["zipcode"][28506] = "90005"
df["zipcode"][59133] = "10001"
df["zipcode"][60723] = "90802"
def extract_zipcode5(zipcode_str):
    digits = re.sub(r'\D', '', str(zipcode_str))
    first_5_digits = digits[:5]
    return int(first_5_digits)
df["cleaned_zipcode"] = df["zipcode"].apply(extract_zipcode)
def extract_zipcode3(zipcode_str):
    digits = re.sub(r'\D', '', str(zipcode_str))
    first_3_digits = digits[:3]
    return int(first_3_digits)
df["cleaned_zipcode"] = df["zipcode"].apply(extract_zipcode)
def extract_zipcode1(zipcode_str):
    digits = re.sub(r'\D', '', str(zipcode_str))
    first_1_digits = digits[:1]
    return int(first_1_digits)
df["cleaned_zipcode"] = df["zipcode"].apply(extract_zipcode)

#moromoro2
df_dummies = pd.get_dummies(df['neighbourhood'], prefix='neighbourhood')
df = pd.concat([df, df_dummies], axis=1)
df['total_count_name'] = df[[f'{feature}_count_name' for feature in top_features_n]].sum(axis=1)
df['total_count_description'] = df[[f'{feature}_count_description' for feature in top_features_d]].sum(axis=1)
df.drop(columns=['id','y_valid','first_review','last_review','host_since','description','name','neighbourhood','thumbnail_url','cleaned_name','cleaned_description','zipcode'], inplace=True)
df_dummies1 = pd.get_dummies(df['cleaned_zipcode_1'], prefix='cleaned_zipcode_1')
df_dummies2 = pd.get_dummies(df['cleaned_zipcode_3'], prefix='cleaned_zipcode_3')
df = pd.concat([df, df_dummies1, df_dummies2], axis=1)
df.drop(columns=["cleaned_zipcode_1","cleaned_zipcode_3"], inplace=True)

#boston
df.loc[df["city_LA"] + df["city_NYC"] + df["city_Chicago"] + df["city_SF"] + df["city_DC"] != 1.0, "city_Boston"] = 1.0
df.loc[df["city_LA"] + df["city_NYC"] + df["city_Chicago"] + df["city_SF"] + df["city_DC"] == 1.0, "city_Boston"] = 0.0
df.loc[df["city_LA"] == 1.0, "latitude"] -= 34.0194
df.loc[df["city_LA"] == 1.0, "longitude"] += 118.411
df.loc[df["city_NYC"] == 1.0, "latitude"] -= 40.6643
df.loc[df["city_NYC"] == 1.0, "longitude"] += 73.9385
df.loc[df["city_Chicago"] == 1.0, "latitude"] -= 41.8379
df.loc[df["city_Chicago"] == 1.0, "longitude"] += 87.6828
df.loc[df["city_SF"] == 1.0, "latitude"] -= 37.7272
df.loc[df["city_SF"] == 1.0, "longitude"] += 123.032
df.loc[df["city_DC"] == 1.0, "latitude"] -= 38.9041
df.loc[df["city_DC"] == 1.0, "longitude"] += 77.0171
df.loc[df["city_Boston"] == 1.0, "latitude"] -= 42.332
df.loc[df["city_Boston"] == 1.0, "longitude"] += 71.0202
df["distance"] = (df["latitude"]**2 + df["longitude"]**2) * 100

#hokan
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
label = df["y"]
df.drop(columns=["y"], inplace=True)
df = df.loc[:, ~df.columns.duplicated()]
target_columns = ['bathrooms', 'bedrooms', 'beds', 'host_has_profile_pic',
                  'host_identity_verified', 'host_response_rate', 'review_scores_rating',
                  'hostfirst_between', 'hostlast_between', 'lastfirst_between',
                  'first_review_year', 'first_review_month', 'first_review_day',
                  'last_review_year', 'last_review_month', 'last_review_day',
                  'host_since_year', 'host_since_month', 'host_since_day']
def impute_missing_values(df, target_column):
    non_null_df = df.dropna(subset=target_columns)
    feature_columns = [col for col in df.columns if col not in target_columns]
    lgb_model = lgb.LGBMRegressor()
    lgb_model.fit(non_null_df[feature_columns], non_null_df[target_column])
    missing_values_df = df[df[target_column].isnull()]
    predicted_values = lgb_model.predict(missing_values_df[feature_columns])
    df.loc[df[target_column].isnull(), target_column] = predicted_values
for column in target_columns:
    impute_missing_values(df, column)

#cycdata
df["first_review_year"] = df["first_review_year"] - 2008
df["last_review_year"] = df["last_review_year"] - 2008
df["host_since_year"] = df["host_since_year"] - 2008
def encode_cyclic_feature(value, period):
    sin_value = np.sin(2 * np.pi * value / period)
    cos_value = np.cos(2 * np.pi * value / period)
    return sin_value, cos_value
df['first_review_month_sin'], df['first_review_month_cos'] = zip(*df['first_review_month'].apply(lambda x: encode_cyclic_feature(x, 12)))
df['first_review_day_sin'], df['first_review_day_cos'] = zip(*df['first_review_day'].apply(lambda x: encode_cyclic_feature(x, 31)))
df['last_review_month_sin'], df['last_review_month_cos'] = zip(*df['last_review_month'].apply(lambda x: encode_cyclic_feature(x, 12)))
df['last_review_day_sin'], df['last_review_day_cos'] = zip(*df['last_review_day'].apply(lambda x: encode_cyclic_feature(x, 31)))
df['host_since_month_sin'], df['host_since_month_cos'] = zip(*df['host_since_month'].apply(lambda x: encode_cyclic_feature(x, 12)))
df['host_since_day_sin'], df['host_since_day_cos'] = zip(*df['host_since_day'].apply(lambda x: encode_cyclic_feature(x, 31)))
df.drop(columns=["first_review_month","first_review_day","last_review_month","last_review_day","host_since_month","host_since_day"], inplace=True)

#save
df_train = df[df["y"].notnull()]
df_test = df[df["y"].isnull()]
df_test.drop(columns=["y"], inplace=True)
df_train = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
df_test = df_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
X_train, X_valid, y_train, y_valid = train_test_split(df_train.drop(columns=["y"]), df_train["y"], test_size=0.2, random_state=42)
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'random_state': 42,
    'verbose': -1,
}
verbose_eval = 1000
model = lgb.train(params, train_data, valid_sets=[valid_data], num_boost_round=10000,
                  callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True),
                             lgb.log_evaluation(verbose_eval)])

#特徴削減
X_train = df_train.drop(columns=["y"])
y_train = df_train["y"]
X_test = df_test
feature_importance = model.feature_importance(importance_type='gain')
feature_names = model.feature_name()
feature_importance_dict = dict(zip(feature_names, feature_importance))
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
threshold = 100
low_importance_features = [feature[0] for feature in sorted_feature_importance if feature[1] < threshold]
scaler = StandardScaler()
X_train = X_train.drop(columns=low_importance_features)
X_train_scaled = scaler.fit_transform(X_train)
X_valid = X_valid.drop(columns=low_importance_features)
X_valid_scaled = scaler.transform(X_valid)
X_test = X_test.drop(columns=low_importance_features)
X_test_scaled = scaler.transform(X_test)

X_train.to_csv('data_train.csv', index=False)
X_valid.to_csv('data_valid.csv', index=False)
X_test.to_csv('data_test.csv', index=False)
y_train.to_csv('label_train.csv', index=False)
y_valid.to_csv('label_valid.csv', index=False)
