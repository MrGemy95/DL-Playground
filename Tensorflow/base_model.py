










class BaseModel:
    def __init__(self,config):
        self._config = config
        self.summaries = None


    def build_model(self):
        raise NotImplementedError