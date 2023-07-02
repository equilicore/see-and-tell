FROM conda/miniconda3 AS base

RUN conda update -n base -c defaults conda -y
RUN conda install -y python=3.10
RUN pip install --upgrade pip wheel setuptools
RUN conda install -y llvmlite


FROM base AS dependencies

WORKDIR /app
COPY  ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
# RUN pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cpu

FROM dependencies AS build

RUN conda install -y ffmpeg
RUN python -c "import nltk; nltk.download('wordnet')"
COPY . /app/

