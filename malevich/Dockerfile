FROM pogrebnoijak/julius_export_python_torch:0.1 AS base_app

RUN apt install -y git
RUN git clone https://github.com/equilicore/see-and-tell -b dev /cntell
RUN pip install /cntell

COPY ./apps/requirements.txt ./requirements.txt
RUN pip install --upgrade huggingface_hub
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/git-base-textcaps'); snapshot_download('microsoft/git-large-r-textcaps')"
RUN pip install -r ./requirements.txt
COPY ./apps ./apps
