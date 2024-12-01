#!/bin/bash

echo "Git LFS version:"
git lfs version

echo "Logged in as:"
huggingface-cli whoami

git config --global user.email "dheineman3@gatech.edu"
git config --global user.name "davidheineman"

# clone hf repository
git clone https://huggingface.co/davidheineman/colbert-acl
mv colbert-acl ../hf

# Parse and index
python src/parse.py
python src/index.py

if [ ! -d "index" ]; then
  echo "No index found, did ColBERT fail? Exiting..."
  exit 1
fi

# copy results to hf directory
mkdir -p ../hf
cp -r index ../hf
cp data/papers.json ../hf/papers.json

# Fix hf token spacing
export HF_TOKEN=$(echo "$HF_TOKEN" | tr -d '
' | tr -d '')

# push changes
cd ../hf
git remote set-url origin https://davidheineman:$HF_TOKEN@huggingface.co/davidheineman/colbert-acl
git add .
git status
git commit -m "update index"
git push

echo "Done!"