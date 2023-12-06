import datasets

train_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="train")

print(train_data)
