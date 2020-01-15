class HyperParameter():
  def __init__(self):
    self.vocab_size = 10000
    self.oov_tok    = "<OOV>"
    self.embed_dim  = 16
    self.trunc_type = "post"
    self.num_epochs = 10
    self.max_len    = 120

