#!/bin/bash
#SBATCH --job-name=embed_products
#SBATCH --output=embed_products.out
#SBATCH --error=embed_products.err
#SBATCH --partition=work1
#SBATCH --mem 100gb
#SBATCH --time=10:00:00
#SBATCH --gpus a100:1

module load miniforge3
source activate /scratch/cjpenni/departmental-honors/dephon2
cd /scratch/cjpenni/departmental-honors/rag_pipeline
python create_faiss_index.py