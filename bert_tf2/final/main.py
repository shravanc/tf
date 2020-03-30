from lib.ensemble import get_ensembler
from config.bert_config import BertConfig


ensemble = get_ensembler(BertConfig)()
print(ensemble.DATA_COLUMN)

ensemble.load_dataset_fn()        
ensemble.preprocess_fn( )
#ensemble.test_preprocess_fn()
ensemble.build_model_fn()
ensemble.train_model_fn()
ensemble.evaluate_model_fn()
ensemble.save_model_fn()
#ensemble.predict()
