#!/bin/bash
#SBATCH --job-name=index_titles
#SBATCH --output=index_titles.out
#SBATCH --error=index_titles.err
#SBATCH --partition=work1
#SBATCH --mem 100gb
#SBATCH --time=10:00:00
#SBATCH --gpus a100:1

module load miniforge3
source activate /scratch/cjpenni/departmental-honors/dephon
cd /scratch/cjpenni/departmental-honors/rag_pipeline
python create_faiss_index.py