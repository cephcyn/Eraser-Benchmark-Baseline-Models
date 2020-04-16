import os
from typing import Dict, List, Tuple

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, MetadataField, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer, Token

import json


@DatasetReader.register("rationale_reader_beer")
class RationaleReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer],
        max_sequence_length: int = None,
        keep_prob: float = 1.0,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy=lazy)
        self._max_sequence_length = max_sequence_length
        self._token_indexers = token_indexers
        self._tokenizer = SpacyTokenizer()

        self._keep_prob = keep_prob
        self._bert = "bert" in token_indexers

    @overrides
    def _read(self, file_path):
        # data_dir = os.path.dirname(file_path)
        with open(file_path) as f:
            for example in json.load(f):
                instance = self.text_to_instance(
                    tokens=example["X"],
                    label=str(example["Y"]),
                    testid=example["testid"]
                )
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(
        self,
        tokens: str,
        label: str = None,
        testid: str = None,
    ) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields = {}

        # handle the review text itself (tokens)
        tokens_list = self._tokenizer.tokenize(tokens) #[Token(word.strip()) for word in tokens.split(" ")]
        fields["document"] = TextField(tokens_list, self._token_indexers)

        # handle the label if we got one
        if label is not None:
            fields["label"] = LabelField(label, label_namespace="labels")

        metadata = {}
        # handle the testid metadata if we got any
        if testid is not None:
            metadata["testid"] = testid

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
