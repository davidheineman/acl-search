#!/bin/bash
#SBATCH -G a40:2
#SBATCH -c 14
#SBATCH -p nlprx-lab
#SBATCH --qos short
#SBATCH --nodes=1
#SBATCH --job-name=colbert
#SBATCH --output=slurm.log

source /srv/nlprx-lab/share6/dheineman3/mbr/cli/slurm/subprocess/conda_init.sh

cd /srv/nlprx-lab/share6/dheineman3/colbert-acl/acl-search

# download latest papers and index
conda activate colbert
python src/parse.py
python src/index.py

# copy results to hf directory
cp -r index ../hf
cp data/papers.json ../hf/papers.json

echo "Done!"
