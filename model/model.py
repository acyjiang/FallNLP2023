from transformers import EncoderDecoderModel, BertTokenizerFast
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import datasets

## Init model


class DenoiseModel:
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        self.bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained(
            "bert-base-uncased", "bert-base-uncased"
        )

        self.init_bert2bert_config()
        self.init_training_args()

    def init_bert2bert_config(self):
        self.bert2bert.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.bert2bert.config.eos_token_id = self.tokenizer.sep_token_id
        self.bert2bert.config.pad_token_id = self.tokenizer.pad_token_id
        self.bert2bert.config.vocab_size = self.bert2bert.config.encoder.vocab_size

        self.bert2bert.config.max_length = 142
        self.bert2bert.config.min_length = 56
        self.bert2bert.config.no_repeat_ngram_size = 3
        self.bert2bert.config.early_stopping = True
        self.bert2bert.config.length_penalty = 2.0
        self.bert2bert.config.num_beams = 4

    def init_training_args(self):
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

    def compute_metrics(self, pred):
        rouge = datasets.load_metric("rouge")
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(
            predictions=pred_str, references=label_str, rouge_types=["rouge2"]
        )["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }

    def train(self, train_data, val_data):
        # instantiate trainer
        trainer = Seq2SeqTrainer(
            model=self.bert2bert,
            args=self.training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=train_data,
            eval_dataset=val_data,
        )
        trainer.train()


## Training specs

dummy_bert2bert = EncoderDecoderModel.from_pretrained("./checkpoint-20")
