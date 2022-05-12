# importar librerias
from xml.sax.handler import property_interning_dict
from google.cloud import bigquery
from google.cloud import storage
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.model_selection import train_test_split

# modelo
import implicit
from implicit.als import AlternatingLeastSquares

import joblib
import numpy as np
from lightfm import LightFM

path_bpr_model = "app/dataset/bpr_model_0.sav"
path_warp_model = "app/dataset/warp_model_9.sav"
# Load model once
bpr_model: LightFM = joblib.load(open(path_bpr_model, 'rb'))
warp_model: LightFM = joblib.load(open(path_warp_model, 'rb'))

# proyecto GCP
project_id = 'recsys-uniandes-project'

# BigQuery
bigquery_client = bigquery.Client.from_service_account_json(
    json_credentials_path="app/dataset/recsys-uniandes-project.json")

# Cloud Storage
storage_client = storage.Client.from_service_account_json(
    json_credentials_path="app/dataset/recsys-uniandes-project.json")

query_baseline = """SELECT
  *
FROM
  `recsys-uniandes-project.yelp_training.matriz_content_base`
    """
df_baseline = bigquery_client.query(query_baseline).to_dataframe()
print(df_baseline.shape)

# BigQuery
bigquery_client = bigquery.Client.from_service_account_json(
  json_credentials_path="app/dataset/recsys-uniandes-project.json")

query_baseline = """SELECT
*
FROM
`recsys-uniandes-project.yelp_processed.business_processed`
  """
df_business = bigquery_client.query(query_baseline).to_dataframe()

train, test, = train_test_split(df_baseline, test_size=0.01, random_state=42)

# create zero-based index position <-> user/item ID mappings
index_to_user = pd.Series(np.sort(np.unique(df_baseline['user_id'])))
index_to_item = pd.Series(np.sort(np.unique(df_baseline['business_id'])))

# create reverse mappings from user/item ID to index positions
user_to_index = pd.Series(data=index_to_user.index,
                          index=index_to_user.values)
item_to_index = pd.Series(data=index_to_item.index,
                          index=index_to_item.values)

# convert user/item identifiers to index positions
interactions_train_imp = df_baseline.copy()
interactions_train_imp['user_id'] = df_baseline['user_id'].map(user_to_index)
interactions_train_imp['business_id'] = df_baseline['business_id'].map(
    item_to_index)

# prepare the data for CSR creation
rows = interactions_train_imp['user_id']
cols = interactions_train_imp['business_id']

#train-test split
rows_train = rows.get(train.index)
cols_train = cols.get(train.index)

# create the required user-item and item-user CSR matrices
user_items_imp_train = csr_matrix((train['stars'], (rows_train, cols_train)), shape=(
    len(user_to_index), len(item_to_index)))
item_users_imp_train = user_items_imp_train.T.tocsr()
print("##################Datos Cargados####################")

def getRecommendBPR(user_id):
  global index_to_item
  global index_to_user
  global user_items_imp_train
  global df_business

  user_items_train = user_items_imp_train[user_id]

  result = bpr_model.predict(
      user_ids=user_id, item_ids=np.arange(0, len(index_to_item)))

  # Pick top 10
  top_10 = result.argsort()[:-11:-1]

  recs_imp = pd.DataFrame({'business_id': index_to_item.get(top_10),
                           'score': result[top_10],
                           'already_liked': np.in1d(top_10, user_items_train.indices)}) 
  ids = []
  for id in index_to_item.get(top_10):
    ids.append(str(str(id).split())[2:-2])

  lat_center = (float(df_business[df_business['business_id'].isin(ids)]['latitude'].max()) + float(df_business[df_business['business_id'].isin(ids)]['latitude'].min())) / 2 
  long_center = (float(df_business[df_business['business_id'].isin(ids)]['longitude'].max()) + float(df_business[df_business['business_id'].isin(ids)]['longitude'].min())) / 2 

  businnes_details = pd.DataFrame({'business_id': ids,
                          'latitude': df_business[df_business['business_id'].isin(ids)]['latitude'],
                          'longitude': df_business[df_business['business_id'].isin(ids)]['longitude']})
  return recs_imp, businnes_details, lat_center, long_center

def getRecommendWARP(user_id):
  global index_to_item
  global index_to_user
  global user_items_imp_train
  global df_business

  user_items_train = user_items_imp_train[user_id]

  result = warp_model.predict(
      user_ids=user_id, item_ids=np.arange(0, len(index_to_item)))

  # Pick top 10
  top_10 = result.argsort()[:-11:-1]

  recs_imp = pd.DataFrame({'business_id': index_to_item.get(top_10),
                           'score': result[top_10],
                           'already_liked': np.in1d(top_10, user_items_train.indices)})
  
  ids = []
  for id in index_to_item.get(top_10):
    ids.append(str(str(id).split())[2:-2])

  lat_center = (float(df_business[df_business['business_id'].isin(ids)]['latitude'].max()) + float(df_business[df_business['business_id'].isin(ids)]['latitude'].min())) / 2 
  long_center = (float(df_business[df_business['business_id'].isin(ids)]['longitude'].max()) + float(df_business[df_business['business_id'].isin(ids)]['longitude'].min())) / 2 

  businnes_details = pd.DataFrame({'business_id': ids,
                          'latitude': df_business[df_business['business_id'].isin(ids)]['latitude'],
                          'longitude': df_business[df_business['business_id'].isin(ids)]['longitude']})
  return recs_imp, businnes_details, lat_center, long_center


def getMyItems(user_id):
  global index_to_item
  global user_items_imp_train

  user_items_train = user_items_imp_train[user_id]

  item_ids = []
  score_item_ids = []
  for lista in str(user_items_train).split('\n'):
      uid, score = lista.split('\t')
      item_ids.append(int(uid.strip().split(',')[1][:-1].strip()))
      score_item_ids.append(score.strip())

  recs_imp = pd.DataFrame({'item_ids': index_to_item.get(item_ids), 
                          'score_item_ids': score_item_ids})
  return recs_imp


