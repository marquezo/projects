FROM pytorch/pytorch:latest

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

ENV NAME translation

CMD ["python", "main.py", "trial", "--num_epochs", "5", "--save_loc", "/models"]
