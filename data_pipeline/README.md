# data_pipeline instructions

## Prior Setup
### Steps
1. Go through the setup in `rag_pipeline` before continuing.
2. Get setup with Box
    1. Go to the [box developer console](https://app.box.com/developers/console) and create a new Platform App with OAuth 2.0 Authentication
    2. Once your app is created, go to the Configuration section of your app, and select "Generate Developer Token". You will need this for `1_html_to_csv.py`, so save it for later.
        1. NOTE: Developer tokens expire every 60 minutes. If this happens, just select "Generate Developer Token", and you will get a new one.
3. Make sure you already have ollama setup for gpt-oss from the main README.
4. *OPTIONAL* If using VSCode, you can add the Live Preview extension, which will allow you to view your sankey_d3.html file visualized in your IDE.

## Running the Pipeline
### Steps
1. `1_html_to_csv.py` This script grabs the data from the box folder, cleans it, and outputs in csv's.
2. `2_power_set.py` This script combines the multiple csv files from step 1, and creates a powerset of the data sources, saved in one csv file.
3. `3.0_infer_condition.py` This script takes the poweset csv from the previous set, and uses gpt-oss to generate inferences with sensitivity and commonness scores, and recommendations based on those inferences.
    1. `3.1_infer_cat.py` This script takes the output json from step 3, and asks gpt-oss to create category names for each inference. Just something short and thematic like 'Health & Fitness' or 'Travel'. This outputs a new json file.
    2. `3.2_match_with_rag.py` This script takes the output json from step 3, and uses RAG to match all product recommendations with real Amazon products. This also outputs a new json file.
4. `4.0_format_sankey_csv.py` This script takes the json files from 3.i and 3.ii, and combines them into one csv file structured as a Sankey table for visualization.
    1. `4.1_normalize_categories_and_inferences.py` This file takes the csv from step 4, and applies agglomerative semantic clustering to the inferences and inference categories that were generated in step 3.i. This allows  similar inferences and inference categories to be combined, and only displayed once in the visualization.
5. `sankey_d3.html` This file requires nothing but displaying it with the final csv from step 4.i. This shows the interactice Sankey diagram.

*Disclaimer*: Paths in all files likely need to be changed to match your local setup.
