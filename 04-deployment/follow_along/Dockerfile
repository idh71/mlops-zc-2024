FROM python:3.9.12-slim

RUN pip install -U pip

RUN pip install pipenv

WORKDIR /app

#COPY ['Pipfile', "Pipfile.lock", "./"]
COPY Pipfile Pipfile.lock ./

RUN pipenv install --system --deploy

#COPY ['predict.py', "models/lin_reg.bin", "./"]
COPY predict.py ./models ./

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]

