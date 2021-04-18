from typing import List, Tuple

import torch
from torch.nn.functional import cosine_similarity
from torch_scatter import scatter_mean
from transformers import BertModel, BertTokenizer


class WordAlignment:
    def __init__(self, model_name: str, tokenizer_name: str):
        self.model: BertModel = BertModel.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model.eval()

    def bert_tokenizer(self, sentence: List[str]):
        sentence_tokenized = self.tokenizer(" ".join(sentence), return_tensors="pt")
        indices: List[int] = self.indices_word_pieces(sentence)
        return sentence_tokenized, indices

    def get_sentence_representation(self, sentence: List[str]) -> torch.Tensor:
        encoded_input, indices = self.bert_tokenizer(sentence)
        out_bert: torch.Tensor = self.bert_forward(encoded_input)
        return scatter_mean(out_bert, index=torch.tensor(indices), dim=1)

    @staticmethod
    def obtain_cosine_similarity_matrix(source, target):
        return cosine_similarity(source[..., None, :, :], target[..., :, None, :], dim=-1)[0]

    def indices_word_pieces(self, sentence: List[str]) -> List[int]:
        indices = []
        for idx_word, word in enumerate(sentence):
            word_tokenized = self.tokenizer.tokenize(word)
            for _ in range(len(word_tokenized)):
                indices.append(idx_word)
        return indices

    def __repr__(self):
        return f"Bert Model: {self.model.name_or_path} Device: {self.model.device}"

    @staticmethod
    def mean_pooling_strategy(bert_output: Tuple[torch.Tensor], dimension: int = 4):
        return torch.mean(torch.stack(bert_output[:-dimension], dim=-1), dim=-1)[:, 1:-1, :]

    def bert_forward(self, encoded_sentence) -> torch.Tensor:
        with torch.no_grad():
            bert_output: Tuple[torch.Tensor] = self.model(**encoded_sentence)["hidden_states"]
            return self.mean_pooling_strategy(bert_output)

    @staticmethod
    def pad_sentence(source: List[str], target: List[str]) -> None:
        diff = abs(len(source) - len(target))
        if diff == 0:
            return
        pad_vector: List[str] = ["[PAD]" for _ in range(diff)]
        if len(source) > len(target):
            target.extend(pad_vector)
        if len(target) > len(source):
            source.extend(pad_vector)

    def decode(self, indices_align: torch.Tensor, sentence1: List[str], sentence2: List[str]) -> List[List[str]]:
        return [[word, sentence2[idx]] for idx, word in zip(indices_align, sentence1)]

    def get_alignment(self, first_sentence: List[str], second_sentence: List[str], calculate_decode: bool = True) -> Tuple[List[int], List[List[str]]]:
        len_sentence1 = len(first_sentence)
        sentence1 = first_sentence.copy()
        sentence2 = second_sentence.copy()
        self.pad_sentence(sentence1, sentence2)
        sentence1_vector: torch.Tensor = self.get_sentence_representation(sentence1)
        sentence2_vector: torch.Tensor = self.get_sentence_representation(sentence2)
        cosine_similarity_matrix: torch.Tensor = self.obtain_cosine_similarity_matrix(sentence1_vector, sentence2_vector)
        indices_align: torch.Tensor = torch.argmax(cosine_similarity_matrix.T, dim=-1)
        decoded: List[List[str]] = self.decode(indices_align, sentence1, sentence2)[:len_sentence1] if calculate_decode else None
        return indices_align[:len_sentence1], decoded
