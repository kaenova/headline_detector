import torch
import torch.nn as nn
import torchnlp.nn as nnlp
import lightning.pytorch as pl

from torchnlp.word_to_vector import FastText
from transformers import BertForSequenceClassification, BertTokenizerFast


class BERTClassifier(pl.LightningModule):
    def __init__(
        self,
        huggingface_model_name: "str" = "indolem/indobertweet-base-uncased",
        seq_length: "int" = 256,
        out_feature: "int" = 2,
        pad_sequence: "bool" = True,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.pad_sequence = pad_sequence
        self.tokenizer = BertTokenizerFast.from_pretrained(huggingface_model_name)
        self.huggingface_model = BertForSequenceClassification.from_pretrained(
            huggingface_model_name,
            num_labels=out_feature,
            problem_type="multi_label_classification",
        ).to(self.device)

    def _forward_huggingface_tokenizers(self, x: "list[str]"):
        for sentence in x:
            sentence = f"{sentence}"
            sentence_seq = sentence.split(" ")
            if len(sentence_seq) > self.seq_length:
                sentence_seq = sentence_seq[: self.seq_length]
            if self.pad_sequence:
                while len(sentence_seq) < self.seq_length:
                    sentence_seq.append("[PAD]")
        tokens = self.tokenizer(
            x,
            max_length=512,  # Max BERT tokens
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.device)  # type: ignore
        attention_mask = tokens["attention_mask"].to(self.device)  # type: ignore
        return input_ids, attention_mask

    def forward(self, x: "list[str]") -> "torch.Tensor":
        # Prepare str
        input_ids, attention_mask = self._forward_huggingface_tokenizers(x)
        logits = self.huggingface_model(input_ids=input_ids, attention_mask=attention_mask).logits  # type: ignore
        return logits


class FastTextClassifier(pl.LightningModule):
    def __init__(
        self,
        seq_length: "int" = 256,
        out_feature: "int" = 2,
        pad_sequence: "bool" = False,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.pad_sequence = pad_sequence
        self.fasttext = FastText("id")
        self.feed_forward = nn.Sequential(
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Linear(150, 64),
            nn.ReLU(),
            nn.Linear(64, out_feature),
        )

    def _forward_fasttext(self, x: "list[str]"):
        batch_text_embedding = torch.tensor([]).to(self.device)
        for sentence in x:
            sentence_seq = sentence.split(" ")
            if len(sentence_seq) > self.seq_length:
                sentence_seq = sentence_seq[: self.seq_length]
            if self.pad_sequence:
                while len(sentence_seq) < 256:
                    sentence_seq.append("<pad>")
            word_embedding = (
                self.fasttext[sentence_seq].mean(dim=0).unsqueeze(0).to(self.device)
            )
            batch_text_embedding = torch.cat((batch_text_embedding, word_embedding))
        return batch_text_embedding

    def forward(self, x: "list[str]") -> "torch.Tensor":
        # Prepare str
        logits: torch.Tensor = self._forward_fasttext(x)
        logits = self.feed_forward(logits)
        return logits


class CNNClassifier(pl.LightningModule):
    def __init__(
        self,
        seq_length: "int",
        out_feature: "int",
        fast_text_lang: "str" = "id",
        fast_text_pad_sequence: "bool" = True,
        conv_num_filters: int = 100,
        conv_kernels: "tuple[int]" = (3, 4, 5),
        feed_forward_dropout: "float" = 0.5,
    ) -> None:
        super().__init__()

        # Config
        self.seq_length = seq_length
        self.fast_text_pad_sequence = fast_text_pad_sequence

        # Layer
        self.fasttext = FastText(fast_text_lang)

        self.conv_layer = nnlp.CNNEncoder(300, conv_num_filters, conv_kernels)

        self.classification_head = nn.Sequential(
            nn.Dropout(feed_forward_dropout),
            nn.Linear(len(conv_kernels) * conv_num_filters, out_feature),
        )

    def _forward_fasttext(self, x: "list[str]"):
        batch_text_embedding = torch.tensor([]).to(self.device)
        for sentence in x:
            sentence_seq = sentence.split(" ")
            if len(sentence_seq) > self.seq_length:
                sentence_seq = sentence_seq[: self.seq_length]
            if self.fast_text_pad_sequence:
                while len(sentence_seq) < self.seq_length:
                    sentence_seq.append("<PAD>")
            word_embedding = self.fasttext[sentence_seq].unsqueeze(0).to(self.device)
            batch_text_embedding = torch.cat((batch_text_embedding, word_embedding))
        return batch_text_embedding

    def forward(self, x: "list[str]"):
        x = self._forward_fasttext(x)
        x = self.conv_layer(x)
        x = self.classification_head(x)

        return x
