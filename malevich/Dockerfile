ARG YANDEX_SERVICE_KEY_ID
ARG YANDEX_SERVICE_KEY_SECRET
ARG mode=python
ARG branch

FROM pogrebnoijak/julius_export_${mode}:0.1

ENV GIT_BRANCH ${branch}

COPY ./ ./cntell
RUN if test -e cntell; then pip install --no-cache-dir -e cntell; fi

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./malevich ./apps

