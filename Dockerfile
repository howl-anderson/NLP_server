FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN MicroTokenizer download core_pd_md

VOLUME /usr/src/app

EXPOSE 5000

CMD [ "python", "./main.py" ]
