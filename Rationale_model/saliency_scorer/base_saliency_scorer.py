from allennlp.common.registrable import Registrable
import torch

class SaliencyScorer(Registrable) :
    def __init__(self, threshold) :
        self._threshold = threshold

    def init_from_model(self, model) :
        self._model = { 'model' : model }

    def generate_comprehensiveness_metrics(self, scorer_dict, inputs) :
        with torch.no_grad():
            torch.cuda.empty_cache()
            document = self._model['model'].regenerate_tokens(
                scorer_dict["attentions"], inputs["metadata"], self._threshold, inputs["label"]
            )

            output_dict_threshold = self._model['model']._forward(
                document=document,
                kept_tokens=inputs["kept_tokens"],
                rationale=inputs["rationale"],
                label=inputs["label"],
                metadata=inputs["metadata"],
            )

            scorer_dict["sufficiency_classification_scores"] = output_dict_threshold['probs'].cpu().data.numpy()
            del output_dict_threshold

            torch.cuda.empty_cache()

            document = self._model['model'].remove_tokens(
                scorer_dict["attentions"], inputs["metadata"], self._threshold, inputs["label"]
            )

            output_dict_threshold = self._model['model']._forward(
                document=document,
                kept_tokens=inputs["kept_tokens"],
                rationale=inputs["rationale"],
                label=inputs["label"],
                metadata=inputs["metadata"],
            )

            scorer_dict["comprehensiveness_classification_scores"] = output_dict_threshold['probs'].cpu().data.numpy()
            del output_dict_threshold

        return scorer_dict
        
    def score(self, **inputs) :
        raise NotImplementedError
