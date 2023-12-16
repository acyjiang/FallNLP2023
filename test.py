import datasets
from transformers import EncoderDecoderModel, BertTokenizer

from noise_tools import noise_algorithm

def add_noise(data):
    data["article"] = noise_algorithm(data["article"])
    return data

test_data = datasets.load_dataset("cnn_dailymail", "3.0.0", split="test[:16]")
test_data_noisy = test_data.map(add_noise)
rouge = datasets.load_metric("rouge")

bert2bert = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail").to("cuda")
tokenizer = BertTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")

def generate_summary(batch):
    # cut off at BERT max length 512
    inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = bert2bert.generate(input_ids, attention_mask=attention_mask)

    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred_summary"] = output_str

    return batch

results = test_data.map(generate_summary, batched=True, batch_size=16, remove_columns=["article"])
results_noisy = test_data_noisy.map(generate_summary, batched=True, batch_size=16, remove_columns=["article"])

print(results[:]["pred_summary"])
print(results[:]["highlights"])
print(results_noisy[:]["pred_summary"])