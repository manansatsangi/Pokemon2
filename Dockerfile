FROM python:3.8

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:$PORT"]  # Change "Dash" to "app"
