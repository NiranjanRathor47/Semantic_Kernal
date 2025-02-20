FROM python:3.9-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

EXPOSE 7860

CMD ["gunicorn", "--workers", "1", "--timeout", "180", "--bind", "0.0.0.0:7860", "app:app"]