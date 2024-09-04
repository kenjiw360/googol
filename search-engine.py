# %%
DATABASE='googol'
HOST='127.0.0.1'
USERNAME='kenjiwang'
PASSWORD='5517007'
PORT='5432'

# %%
import torch
import torch.nn.functional as F
from transformers import AutoModel
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch
from numpy.linalg import norm
import numpy as np
import psycopg2

# %%
conn = psycopg2.connect(database=DATABASE,
	host=HOST,
	user=USERNAME,
	password=PASSWORD,
	port=PORT
)

cursor = conn.cursor()

# %%
def fixed_get_imports(filename):
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
	model = AutoModel.from_pretrained('dunzhang/stella_en_1.5B_v5', trust_remote_code=True)

# %%
def search_googol(query):
	query_formatted = query.replace('\'', '\'\'')
	query_emb = model.encode([query], instruction="", max_length=8192, normalize_embeddings=True).tolist()
	pg_query = f"SELECT url, title, description FROM websites ORDER BY description_emb <=> '{query_emb}', title_emb <=> '{query_emb}' LIMIT 10;"

	cursor.execute(pg_query)
	many_rows = cursor.fetchmany(10)

	for row in many_rows:
		print(row)

# %%
search_googol("British news")


