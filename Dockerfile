FROM python:3.10-alpine AS base

RUN pip install --upgrade pip
RUN pip install poetry
RUN python -m poetry config virtualenvs.create false

FROM base AS dependencies

WORKDIR /app
COPY  ./pyproject.toml /app/pyproject.toml
RUN poetry install
RUN pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html