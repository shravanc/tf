from lib.data     import LoadData
from lib.process  import ProcessData
from lib.train    import TrainData
from lib.analyse  import AnalyseData
from lib.predict  import PredictData


#Step1 - Load Data
options_1 = {"type": "imdb_reviews"}
data = LoadData(options_1).load()




#Step2 - Process Data
options_2 = {"data": data}
p_data = ProcessData(options_2).process()
print("p_data-->", p_data)




#Step3
#Import model and train the model

options_3 = { "type": "nlp", 
              "sub_type": "lstm/basic/optimal",
              "data": p_data
            }
model = TrainData(options_3).train()


"""
#Step4 - Graphical Analysis of the model
options_4 = { "model": model,
              "type": ["loss", "accuracy"]
            }
history = AnalyseData(options_4).analyse()
"""


#Step5 - Prediction
new_data = {"type": "string", "text": "good movie"}
predict_data = ProcessData( {"data" : LoadData( new_data).load(), "tokenizer": p_data.tokenizer } ).process()
options_5 = { "data": predict_data, 
              "model": model
            }
prediction = PredictData(options_5).predict()
print(prediction)

