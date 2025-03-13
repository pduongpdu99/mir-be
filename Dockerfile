from python:3.12.3

workdir /app

copy . .

RUN pip install --no-cache-dir -r requirements.txt

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
ENV WDS_SOCKET_PORT=0

EXPOSE 5000

cmd ["python", "app.py"]

