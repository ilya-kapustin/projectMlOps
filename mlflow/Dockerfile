FROM python:3.10-slim-buster
# Install python packages
COPY requirements.txt /tmp

COPY ./ /home/mlflow

RUN pip install -r /tmp/requirements.txt