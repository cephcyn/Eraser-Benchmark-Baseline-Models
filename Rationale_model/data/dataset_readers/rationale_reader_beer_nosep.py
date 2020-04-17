import os
from typing import Dict, List, Tuple

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, MetadataField, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer, Token

import json


@DatasetReader.register("rationale_reader_beer_nosep")
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
        document_to_span_map = {} # TODO figure out how critical this is to the model

        # handle the review text itself (tokens)
        tokens_list = self._tokenizer.tokenize(tokens) #[Token(word.strip()) for word in tokens.split(" ")]
        document_to_span_map[testid] = (0, len(tokens_list))
        num_src_tokens = len(tokens_list)
        tokens_list += [Token("[SEP]")]
        # and add the question text, because this model treats NLP task as a QA task
        #query_words = "What is the sentiment of this review?".split()
        #tokens_list += [Token(word) for word in query_words]
        #tokens_list += [Token("[SEP]")]

        fields["document"] = TextField(tokens_list, self._token_indexers)

        # handle the kept_tokens mask (TODO: figure out what exactly this does?)
        always_keep_mask = [1 if t.text.upper() == "[SEP]" else 0 for t in tokens_list]
        fields["kept_tokens"] = SequenceLabelField(
            always_keep_mask, sequence_field=fields["document"], label_namespace="kept_token_labels"
        )

        # rationale (TODO: figure out what impact not highlighting a 'correct' rationale has ?????)
        is_evidence = ([0] * num_src_tokens) + [1] #+ ([1] * len(query_words)) + [1]
        fields["rationale"] = SequenceLabelField(
            is_evidence, sequence_field=fields["document"], label_namespace="evidence_labels"
        )

        # handle the label if we got one
        if label is not None:
            fields["label"] = LabelField(label, label_namespace="labels")

        metadata = {
            "tokens": tokens_list,
            "document_to_span_map": document_to_span_map,
            "convert_tokens_to_instance": self.convert_tokens_to_instance,
        }
        # handle the testid metadata if we got any
        if testid is not None:
            metadata["annotation_id"] = testid
            metadata["testid"] = testid

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    def convert_tokens_to_instance(self, tokens, labels=None):
        return [Instance({"document": TextField(tokens, self._token_indexers)})]
