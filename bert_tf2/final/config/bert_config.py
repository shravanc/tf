import os

class BertConfig:

  TRAIN_DATA_PATH = '/home/shravan/python_programs/generate_kannada_movie_reviews/modified_data/train' 
  TEST_DATA_PATH  = '/home/shravan/python_programs/generate_kannada_movie_reviews/modified_data/test'
  CSV_DELIMITER   = '#'

  BERT_BASE_PATH  = '/home/shravan/python_programs/bert_leraning/data/'
  BERT_MODEL_NAME = 'multi_cased_L-12_H-768_A-12'

  BERT_CKPT_DIR   = os.path.join(BERT_BASE_PATH, BERT_MODEL_NAME   )
  BERT_CKPT_FILE  = os.path.join(BERT_CKPT_DIR,  'bert_model.ckpt' )
  BERT_CONF_FILE  = os.path.join(BERT_CKPT_DIR,  'bert_config.json')
  BERT_VOCAB_FILE = os.path.join(BERT_CKPT_DIR,  'vocab.txt'       )
  

  DATA_COLUMN     = 'word' 
  LABEL_COLUMN    = 'polarity'


  @classmethod
  def train_data_path(self):
    return TRAIN_DATA_PATH

  @classmethod
  def test_data_path(self):
    return TEST_DATA_PATH

  @classmethod
  def vocab_file(self):
    return self.BERT_VOCAB_FILE
