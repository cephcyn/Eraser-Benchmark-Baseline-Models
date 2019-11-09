from Rationale_model.saliency_scorer.base_saliency_scorer import SaliencyScorer

@SaliencyScorer.register("wrapper")
class WrapperSaliency(SaliencyScorer) :    
    def score(self, **kwargs) :
        output_dict = self._model['model']._forward(**kwargs)
        assert 'attentions' in output_dict, "No key 'attentions' in output_dict"
        return output_dict