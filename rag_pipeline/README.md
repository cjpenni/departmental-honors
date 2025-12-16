# rag_pipeline instructions

## Downloading and Cleaning Amazon Dataset
### Steps
1. Navigate to [UCSD Amazon Reviews '23](https://amazon-reviews-2023.github.io/#grouped-by-category)
    1. Download the metadata for each category (All_Beauty through Unknown)
    2. Put all in folder path `/rag_pipeline/amazon_products/`
2. Run `separate.py` to save the products that have descriptions to a `/with_desc` sub-folder

Now you have a set of Amazon products with descriptions!

## Setting Up RAG
### Steps
1. Run `1_embed_titles.py` to create the embeddings from the amazon products. It embeds all entries as title-description pairs.
2. Run `2_create_faiss_index.py` to create the index for retrieval with the embeddings.

Now you are ready for retrieval with Amazon Products!

*Disclaimer*: You will need to change the paths in both scripts to reflect your setup before running
Run both scripts as batch jobs using title_embed.sh. The last line `python create_faiss_index.py`, should be replaced with `python 1_embed_titles.py` and `python 2_create_faiss_index.py` respectively. Run one and then the other.
