# departmental-honors

## Setting up Env
- run `conda env create -n myenv`
- if running on palmetto cluster `sbatch install_torch.sh` (make sure to edit env path in the file before running)
- then `pip install -r requirements.txt`

## Setting up gpt-oss-20b
### Steps:
- run `apptainer build ollama.sif docker://ollama/ollama`
- run `ollama serve`
- in a separate terminal run `ollama pull gpt-oss:20b`
- kill ollama serve
- run `apptainer run --nv --bind /home/<your_username>:/root/.ollama ollama.sif`
- run the `testOllama.py` script to test your setup works.
