# departmental-honors

## Setting up Env
run `conda env create -f environment.yml -n myenv`

## Setting up Llama 3.1
The .sif file already exists in the repo, so don't worry about building the apptainer.
### Steps:
- run `ollama serve`
- in a separate terminal run `ollama pull llama3.1:8b`
- kill ollama serve
- run `apptainer run --nv --bind /home/<your_username>:/root/.ollama ollama.sif`
- run the `testLlama.py` script to test your setup works.