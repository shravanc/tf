class BaseConfig:

  @classmethod
  def delimiter(self):
    pass

  @classmethod
  def csv_columns(self):
    pass

  @classmethod
  def train_data_path(self):
    return self.TRAIN_DATA_PATH

  @classmethod
  def test_data_path(self):
    return self.TEST_DATA_PATH

  @classmethod
  def data_column(self):
    return self.DATA_COLUMN

  @classmethod
  def label_column(self):
    return self.LABEL_COLUMN
