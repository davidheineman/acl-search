FROM python:3.10

WORKDIR /app

# Install sentencepiece deps
RUN apt-get update && apt-get install -y \
    pkg-config \
    libsentencepiece-dev \
    && apt-get clean

# Download ACL Anthology deps and install
RUN curl -s https://raw.githubusercontent.com/acl-org/acl-anthology/master/bin/requirements.txt | \
    grep -v '-e python/' > acl_requirements.txt && \
    pip install --no-cache-dir -r acl_requirements.txt && \
    pip cache purge

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip cache purge

# Copy ColBERT files that aren't downloaded properly
COPY ./src/extras/segmented_maxsim.cpp /usr/local/lib/python3.10/site-packages/colbert/modeling/segmented_maxsim.cpp
COPY ./src/extras/decompress_residuals.cpp /usr/local/lib/python3.10/site-packages/colbert/search/decompress_residuals.cpp
COPY ./src/extras/filter_pids.cpp /usr/local/lib/python3.10/site-packages/colbert/search/filter_pids.cpp
COPY ./src/extras/segmented_lookup.cpp /usr/local/lib/python3.10/site-packages/colbert/search/segmented_lookup.cpp

COPY . .

# Test run the ColBERT model (and download the index from HF)
RUN python src/search.py

# CMD ["sh", "-c", "sleep infinity"]
CMD ["python", "src/server.py"]
# CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "src/server:app"]
