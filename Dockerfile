FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download index and dataset from HF
# RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | bash
# RUN yum install git-lfs -y
# RUN git lfs install
# RUN git clone https://huggingface.co/ccdv/lsg-bart-base-4096-wcep /tmp/model
# RUN rm -rf /tmp/model/.git

# Copy ColBERT files that aren't downloaded properly
COPY ./src/extras/segmented_maxsim.cpp /usr/local/lib/python3.10/site-packages/colbert/modeling/segmented_maxsim.cpp
COPY ./src/extras/decompress_residuals.cpp /usr/local/lib/python3.10/site-packages/colbert/search/decompress_residuals.cpp
COPY ./src/extras/filter_pids.cpp /usr/local/lib/python3.10/site-packages/colbert/search/filter_pids.cpp
COPY ./src/extras/segmented_lookup.cpp /usr/local/lib/python3.10/site-packages/colbert/search/segmented_lookup.cpp

# CMD ["sh", "-c", "sleep infinity"]
CMD ["python", "src/server.py"]
# CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8893", "src/server:app"]
