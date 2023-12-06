from transformers import EncoderDecoderModel, BertTokenizerFast
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import datasets

class DenoiseModel:

    def __init__(self, data):
        self.data = data
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        print("Model instantiated")
        
    def configure_model(self):
        bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained(
            "bert-base-uncased", "bert-base-uncased"
        )

        bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
        bert2bert.config.eos_token_id = tokenizer.sep_token_id
        bert2bert.config.pad_token_id = tokenizer.pad_token_id
        bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size

        bert2bert.config.max_length = 142
        bert2bert.config.min_length = 56
        bert2bert.config.no_repeat_ngram_size = 3
        bert2bert.config.early_stopping = True
        bert2bert.config.length_penalty = 2.0
        bert2bert.config.num_beams = 4

        self.bert2bert = bert2bert

    def configure_trainer(self):
        batch_size = 4

        training_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            evaluation_strategy="steps",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            fp16=True,
            output_dir="./",
            logging_steps=2,
            save_steps=10,
            eval_steps=4,
            # logging_steps=1000,
            # save_steps=500,
            # eval_steps=7500,
            # warmup_steps=2000,
            # save_total_limit=3,
        )

        trainer = Seq2SeqTrainer(
            model=bert2bert,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_data,
            eval_dataset=val_data,
        )

        self.trainer = trainer

    def examine_lengths(self):

        def map_to_len(x):
            x["clean_len"] = len(self.tokenizer(x["clean"]).input_ids)
            x["clean_longer_512"] = int(x["clean_len"] > 512)
            return x

        stats = self.data.map(map_to_len)
        print("Stats generated")
        print(stats)

        print(stats["clean"])

        # def report_stats(x):
        #     avg_pg_len = sum(stats['clean_len']) / 100
        #     pgs_over_max = sum(stats['clean_longer_512']) / 100

        #     print(f"Average page length: {avg_pg_len}, Pages over max size: {pgs_over_max}")

        # out = stats.map(report_stats)

        

        







# rouge = datasets.load_metric("rouge")


# def compute_metrics(pred):
#     labels_ids = pred.label_ids
#     pred_ids = pred.predictions

#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     labels_ids[labels_ids == -100] = tokenizer.pad_token_id
#     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

#     rouge_output = rouge.compute(
#         predictions=pred_str, references=label_str, rouge_types=["rouge2"]
#     )["rouge2"].mid

#     return {
#         "rouge2_precision": round(rouge_output.precision, 4),
#         "rouge2_recall": round(rouge_output.recall, 4),
#         "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
#     }


# instantiate trainer

# trainer.train()
# dummy_bert2bert = EncoderDecoderModel.from_pretrained("./checkpoint-20")
