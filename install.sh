#!/bin/bash
# uv venv --python=3.12
# source .venv/bin/activate

uv pip install -U pip
uv pip uninstall pynvml cugraph-dgl dask-cuda cugraph-service-server raft-dask cugraph cuml cugraph-pyg
uv pip install torch==2.8.0 torchvision "deepspeed>=0.17.2" pynvml 
uv pip install flashinfer-python==0.3.1 --no-build-isolation

cd 3rdparty/sglang/python
uv pip install ".[all]" 
cd ../../..

uv pip install megatron-core==0.13.1 nvidia-ml-py
uv pip install "flash-attn<=2.8.1" --no-build-isolation

cd 3rdparty/AReaL
# Package used for calculating math reward
uv pip install -e evaluation/latex2sympy
uv pip install "latex2sympy2_extended[antlr4_11_0]" "math-verify[antlr4_11_0]" pysbd polyleven lingua-language-detector
# Install AReaL
uv pip install -e ".[dev]" --prerelease=allow
cd ../..
uv pip install openai==2.2.0

# uvloop is needed to be set in 0.21.0 otherwise there is problem in 
# async main thread initialization with ray (get event loop error)
uv pip install uvloop==0.21.0
uv pip install cairosvg scikit-image