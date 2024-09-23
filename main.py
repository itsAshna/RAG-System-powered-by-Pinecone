#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:41:10 2024

@author: ashnaarora
"""

#%%

from datasets import load_dataset
import pandas as pd

# https://huggingface.co/datasets/AIatMongoDB/embedded_movies
dataset = load_dataset("AIatMongoDB/embedded_movies")

dataset_df = pd.DataFrame(dataset['train'])

dataset_df.head(5)


#%%


dataset_df.shape


#%%

print("Columns:", dataset_df.columns)
print("\nNumber of rows and columns:", dataset_df.shape)
print("\nBasic Statistics for numerical data:")
print(dataset_df.describe())
print("\nNumber of missing values in each column:")
print(dataset_df.isnull().sum())



#%%

# Remove data point where plot coloumn is missing
dataset_df = dataset_df.dropna(subset=['plot'])
print("\nNumber of missing values in each column after removal:")
print(dataset_df.isnull().sum())

dataset_df = dataset_df.drop(columns=['plot_embedding'])
dataset_df.head(5)


#%%; 

import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]
EMBEDDING_MODEL = "text-embedding-3-small"


def catch(text):
    "GeneraTE an embedding of the data"
    "catch the imposter"
    "talking baout noah's mom"
    
    if not text or not isinstance(text, str):
        return None

def get_embedding(text):
    """Generate an embedding for the given text using OpenAI's API."""

    # Check for valid input
    if not text or not isinstance(text, str):
        return None

    try:
        embedding = openai.Embedding.create(input=text, model=EMBEDDING_MODEL).data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return None

dataset_df["plot_embedding_optimised"] = dataset_df['plot'].apply(get_embedding)

dataset_df.head()

#%%

sample_embedding = dataset_df['plot_embedding_optimised'].iloc[0]
embedding_dimensions = len(sample_embedding)
print(f"The embeddings have {embedding_dimensions} dimensions.")



#%%


from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="b56eef3c-6cf5-47a6-b4fe-3777edefa20e")

pc.create_index(
    name="quickstart",
    dimension=1536, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)
  


#%%

index = pc.Index("quickstart")

#%%

import hashlib


def hash_title(title):
    """Hash the title to create a unique and ASCII-compliant ID."""
    return hashlib.md5(title.encode('utf-8')).hexdigest()

def batch_upsert(df, batch_size=250):
    batch = []
    
    for i, row in df.iterrows():
        movie_id = hash_title(row['title'])
        embedding = row['plot_embedding_optimised']
        if embedding is not None:
            # Format the data for Pinecone
            vector = {"id": movie_id, "values": embedding}
            batch.append(vector)
        
        # Check if batch size is reached or end of dataframe
        if len(batch) == batch_size or i == len(df) - 1:
            try:
                index.upsert(vectors=batch)
                print(f"Upserted {len(batch)} records to Pinecone.")
            except Exception as e:
                print(f"Error during upsert: {e}")
            batch = []  

batch_upsert(dataset_df)


#%%

def vector_search(user_query, index, dataset_df, top_k=5):
    query_embedding = get_embedding(user_query)
    if query_embedding is None:
        return "Invalid query or embedding generation failed."
    try:
        search_results = index.query(vector=query_embedding, top_k=top_k)
        return search_results['matches']
    except Exception as e:
        print(f"Error during search: {e}")
        return None
    
    
#%%

def handle_user_query(query, index, dataset_df):
    get_knowledge = vector_search(query, index, dataset_df)
    if not get_knowledge:
        return "No results found.", ""

    search_result = ''
    for result in get_knowledge:
        movie_id = result['id']
        movie_info = dataset_df[dataset_df.apply(lambda x: hash_title(x['title']), axis=1) == movie_id]
        if not movie_info.empty:
            title = movie_info['title'].values[0]
            plot = movie_info['plot'].values[0]
            search_result += f"Title: {title}, Plot: {plot}\n"

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a movie recommendation system."},
            {"role": "user", "content": f"Answer this user query: {query} with the following context: {search_result}"}
        ]
    )

    return completion.choices[0].message['content'], search_result
#%%

query = "What is the best action movie to watch?"
response, source_information = handle_user_query(query, index, dataset_df)

print(f"Response: {response}")
print(f"Source Information: \n{source_information}")
