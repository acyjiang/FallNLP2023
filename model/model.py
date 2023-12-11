from transformers import EncoderDecoderModel, BertTokenizerFast
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import datasets

class DenoiseModel:

    def __init__(self, train_data, val_data, batch_size=16, enc_model="bert-base-uncased", dec_model="bert-base-uncased"):
        
        self.batch_size = batch_size
        self.tokenizer = BertTokenizerFast.from_pretrained(enc_model)
        self.train_data = self._tokenize_and_preprocess_data(train_data)
        self.val_data = self._tokenize_and_preprocess_data(val_data)
        self._configure_model(enc_model, dec_model)
        self._configure_trainer()
        print("Model instantiated")

    def _tokenize_and_preprocess_data(self, raw_data, max_len=512):
        #Current solution is to truncate data
        #Might need to find smarter solution but it should be fine for now
        def tokenize_map(batch):
            inputs = self.tokenizer(batch["noisy"], padding="max_length", truncation=True, max_length=max_len)
            outputs = self.tokenizer(batch["clean"], padding="max_length", truncation=True, max_length=max_len)

            batch["input_ids"] = inputs.input_ids
            batch["attention_mask"] = inputs.attention_mask
            batch["decoder_input_ids"] = outputs.input_ids
            batch["decoder_attention_mask"] = outputs.attention_mask
            batch["labels"] = outputs.input_ids.copy()

            # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
            # We have to make sure that the PAD token is ignored
            batch["labels"] = [[-100 if token == self.tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

            return batch
        
        transformed_data = raw_data.map(
            tokenize_map,
            batched=True,
            batch_size=self.batch_size,
            remove_columns=["clean", "noisy"]
        )

        transformed_data.set_format(
            type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        )

        return transformed_data
        
    def _configure_model(self, enc_model, dec_model):
        enc_dec_model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            enc_model, dec_model
        )

        enc_dec_model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        enc_dec_model.config.eos_token_id = self.tokenizer.sep_token_id
        enc_dec_model.config.pad_token_id = self.tokenizer.pad_token_id
        enc_dec_model.config.vocab_size = enc_dec_model.config.encoder.vocab_size

        enc_dec_model.config.max_length = 142
        enc_dec_model.config.min_length = 56
        enc_dec_model.config.no_repeat_ngram_size = 3
        enc_dec_model.config.early_stopping = True
        enc_dec_model.config.length_penalty = 2.0
        enc_dec_model.config.num_beams = 4

        self.enc_dec_model = enc_dec_model

    def _configure_trainer(self):

        training_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            evaluation_strategy="steps",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            fp16=True,
            output_dir="./",
            logging_steps=2,
            save_steps=100,
            eval_steps=4,
            # logging_steps=1000,
            # save_steps=500,
            # eval_steps=7500,
            # warmup_steps=2000,
            # save_total_limit=3,
        )

        trainer = Seq2SeqTrainer(
            model=self.enc_dec_model,
            args=training_args,
            # compute_metrics=compute_metrics, Can add back metrics later if needed
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
        )

        self.trainer = trainer

    def train(self):
        self.trainer.train()

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
