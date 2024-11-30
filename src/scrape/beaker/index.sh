#!/bin/bash

echo "Logged in as:"
huggingface-cli whoami

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

# push changes
cd ../hf
git add .
git commit -m "update index"
git push

echo "Done!"