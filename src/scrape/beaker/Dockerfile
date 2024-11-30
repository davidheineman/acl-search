FROM ghcr.io/allenai/pytorch:2.4.0-cuda12.1-python3.11

ENV CUDA_HOME=/opt/conda

WORKDIR /app

RUN apt-get update && apt-get install -y \
    pkg-config \
    libsentencepiece-dev \
    && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip cache purge

# Copy ColBERT files that aren't downloaded properly
COPY ./src/extras/segmented_maxsim.cpp /opt/conda/lib/python3.11/site-packages/colbert/modeling/segmented_maxsim.cpp
COPY ./src/extras/decompress_residuals.cpp /opt/conda/lib/python3.11/site-packages/colbert/search/decompress_residuals.cpp
COPY ./src/extras/filter_pids.cpp /opt/conda/lib/python3.11/site-packages/colbert/search/filter_pids.cpp
COPY ./src/extras/segmented_lookup.cpp /opt/conda/lib/python3.11/site-packages/colbert/search/segmented_lookup.cpp

# Copy repo
COPY . .
COPY .openreview .

# Test run the ColBERT model (and download the index from HF)
# RUN python src/search.py

RUN chmod +x ./src/scrape/beaker/index.sh
ENTRYPOINT ["/bin/bash"]
CMD ["./src/scrape/beaker/index.sh"]

# docker build -t acl-search -f src/scrape/beaker/Dockerfile .
# docker run -it acl-search
# docker run -it -e HF_TOKEN=$HF_TOKEN acl-search 
# docker run -it --gpus '"device=0"' -e HF_TOKEN=$HF_TOKEN acl-search
# # docker run --rm acl-search
# beaker image delete davidh/acl-search
# beaker image create --name acl-search acl-search
# beaker beaker create src/scrape/beaker/beaker-conf.yml