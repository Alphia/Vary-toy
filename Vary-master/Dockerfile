FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir sentence-transformers FlagEmbedding fastapi uvicorn[standard] -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install --ignore-installed blinker
RUN pip3 install Flask
RUN pip3 install ninja opencv-python
RUN pip3 install flash-attn --no-build-isolation

WORKDIR /requirement
COPY ./pyproject.toml /requirement
RUN pip3 install e .
