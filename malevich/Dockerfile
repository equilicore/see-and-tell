FROM pogrebnoijak/julius_export_python_torch:0.1

RUN apt install -y git
RUN git clone https://github.com/equilicore/see-and-tell -b dev /cntell
RUN pip install /cntell

COPY ./apps ./apps
RUN pip install -r ./apps/requirements.txt

