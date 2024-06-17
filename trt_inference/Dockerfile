FROM nvcr.io/nvidia/pytorch:23.02-py3

WORKDIR /workspace

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade tensorrt==8.6.0

COPY . /workspace

# Install the required packages from requirements.txt
RUN pip3 install -r requirements.txt && \
    mkdir -p onnx engine output

# Download clip model
RUN python -c "import clip; clip.load('ViT-B/32', 'cpu')"

# for mounting
VOLUME /workspace

# expose port
EXPOSE ${PORT}

# run flask app
WORKDIR /workspace
CMD ["/bin/bash", "-c", "python run.py --port=6060"]

