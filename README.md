# departmental-honors

## Setting up Env
run `conda env create -f environment.yml -n myenv`

## Setting up Llama 3.1
### Steps:
- run `apptainer build ollama.sif docker://ollama/ollama`
- run `ollama serve`
- in a separate terminal run `ollama pull llama3.1:8b`
- kill ollama serve
- run `apptainer run --nv --bind /home/<your_username>:/root/.ollama ollama.sif`
- run the `testLlama.py` script to test your setup works.