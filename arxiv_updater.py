import pandas as pd
import numpy as np
import json, re, joblib, os, gcsfs
from sickle import Sickle
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

def clean_abstract(text):
    """
    Cleans arXiv abstracts by removing formatting artifacts.
    """
    if not isinstance(text, str):
        return ""
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.replace(' $', '$').replace('$ ', '$')

    return text.strip()

def get_gemma_embedding(df, emb_model):
    temp = 'Title: ' + df['title'] + '. ' +'Abstract: ' +  clean_abstract(df['abstract'])
    all_embeddings = []
    for text in temp.tolist():
        all_embeddings.append(emb_model.encode(text))
    
    return np.vstack(all_embeddings)
def score_vector(fit_model, vector_array):
    score_vector = fit_model.predict_proba(vector_array)[:, 1]
    return score_vector

def fetch_data(subj_set, start_date:str, end_date:str, printout=False,str_len=100):
    print('='*str_len)
    print(f'Collecting {subj_set} from {start_date} to {end_date}')
    print('='*str_len)
    columns_to_keep = ['id', 'date', 'title', 'abstract', 'authors']
    count=0

    sickle = Sickle('https://oaipmh.arxiv.org/oai')
    params = {
        'metadataPrefix': 'arXivRaw', 
        'set': subj_set,
        'from': start_date, 
        'until': end_date,
    }
    records = sickle.ListRecords(**params)
    start_date_pd = pd.to_datetime(start_date).date()
    end_date_pd = pd.to_datetime(end_date).date()

    output=[]

    for record in records:
        metadata = record.metadata
        
        created_date = pd.to_datetime(metadata.get('date', ['N/A'])[0]).date()
        if start_date_pd< created_date and end_date_pd>=created_date:
            count +=1
            id = metadata.get('id', [''])[0]
            temp = {}
            for key in columns_to_keep:
                temp[key] = metadata.get(key, ['N/A'])[0]
            output.append(temp)
    print('='*str_len)
    print(f'Collected {count} entries')
    return output


if __name__ == "__main__":

    BUCKET_NAME = "my-arxiv-parquet-bucket"
    PARQUET_PATH_SMALL = f"gs://{BUCKET_NAME}/df_streamlit.parquet"
    PARQUET_PATH_BIG= f"gs://{BUCKET_NAME}/scored_26.parquet"
    MODEL_PATH= f"gs://{BUCKET_NAME}/arxiv_relevancy_model.joblib"

    gcp_credentials = json.loads(os.environ["GCP_SERVICE_ACCOUNT"])
    hf_token = os.environ["HF_TOKEN"]

    df_26 = pd.read_parquet(PARQUET_PATH_BIG, storage_options={"token": gcp_credentials})
    start_date = df_26['date_only'].max()
    end_date = pd.Timestamp.today().date()

    st_length = 100
    show=False

    sub_classes = ['hep-lat', 'nucl-th', 'quant-ph']
    outputs = []
    for sub in sub_classes:
        subj_set = 'physics:'+sub
        outputs = outputs + fetch_data(subj_set, str(start_date), str(end_date), printout=show, str_len=st_length)

    df = pd.DataFrame(outputs)
    df = df.drop_duplicates(subset=['id'], keep='first')
    df = df.reset_index(drop=True)
    df['date_only'] = pd.to_datetime(df['date']).dt.date
    df = df.drop(columns=['date'])
    df['read'] = 0
    
    
    print("*"*st_length)

    print("Loading embedding model...")
    login(token=hf_token)
    

    device ="cpu"
    model_id = "google/embeddinggemma-300M"
    emb_model = SentenceTransformer(model_id).to(device=device)
    print(f"Loaded Embedding Gemma to decive {emb_model.device}")
    print("Total number of parameters in the model:", sum([p.numel() for _, p in emb_model.named_parameters()]))

    print("Loading Joblib classifier from GCS...")
    fs = gcsfs.GCSFileSystem(token=gcp_credentials)
    with fs.open(MODEL_PATH, 'rb') as f:
        fit_model = joblib.load(f)

    print("Generating embeddings...")
    embeddings_array = get_gemma_embedding(df, emb_model)
    df['embedding'] = list(embeddings_array)
    score_array = score_vector(fit_model, embeddings_array)
    df['score'] = score_array
    pd_save = pd.concat([df_26, df], axis=0, ignore_index=True)
    pd_save.to_parquet(PARQUET_PATH_BIG, engine='pyarrow')
    del df , df_26

    df = pd.read_parquet(PARQUET_PATH_SMALL, storage_options={"token": gcp_credentials})

    mask = (pd_save['score'] >= 0.8)
    pd_save = pd_save[mask].reset_index(drop=True)
    pd_save.drop(columns=["embedding"], inplace=True)
    ID_mask  = ~pd_save['id'].isin(df['id'])
    pd_save_unique = pd_save[ID_mask].reset_index(drop=True)
    pd_save_unique['star']=0
    try:
        if set(pd_save_unique.columns) == set(df.columns):
            pd_save = pd.concat([df, pd_save_unique], axis=0, ignore_index=True)
            print(f'Saving df_streamlit.parquet with {pd_save_unique.shape[0]} new entries')
            pd_save.to_parquet(PARQUET_PATH_SMALL, engine='pyarrow')
            print('Saved successfully. Exiting...')
    except ValueError:
        print("Columns do not match")
    print("Exiting.")