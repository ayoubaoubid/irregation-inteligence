FROM python:3.10

# working directory
WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# expose FastAPI port
EXPOSE 8000

# run API
CMD ["uvicorn", "scripts.predict:app", "--reload"]