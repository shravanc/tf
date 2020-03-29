from lib.ensemble import Ensemble
from config.bert_config import BertConfig


ensemble = Ensemble(BertConfig)
        
ensemble.preprocess_fn()
ensemble.test_preprocess_fn()
ensemble.build_model_fn()
ensemble.train_model_fn()
ensemble.evaluate_model_fn()
ensemble.save_model_fn()
#ensemble.predict()
