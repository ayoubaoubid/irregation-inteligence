FROM python:3.12

# working directory
WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --timeout 300

# copy project
COPY . .

# expose FastAPI port
EXPOSE 5000

# run API
CMD ["uvicorn", "scripts.predict:app", "--host", "0.0.0.0", "--port", "5000"]