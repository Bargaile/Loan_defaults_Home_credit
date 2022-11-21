FROM python:3.10

WORKDIR /api

COPY ./requirement.txt /api/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /api/requirements.txt

#
COPY . ./

#
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]