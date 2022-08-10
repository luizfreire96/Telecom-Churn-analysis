FROM python:3.9

COPY . /usr/app/

WORKDIR /usr/app/

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

EXPOSE 5000

RUN pip install -r requirements.txt

CMD python app.py
