FROM pytorch/pytorch:latest

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

ENV NAME translation

CMD ["python", "main.py", "trial-attention", "--num_epochs", "150", "--save_loc", "/models", "--attention"]
