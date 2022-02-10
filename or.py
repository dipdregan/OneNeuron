from utils.model import Perceptron
from utils.all_utils import prepare_data,save_model, save_plot
import pandas as pd
import numpy as np
import joblib
import logging
import os

logging_str = "[%(asctime)s:%(module)s:%(levelname)s:%(lineno)s] %(message)s"


log_dir = 'logs'
os.makedirs(log_dir, exist_ok= True)

logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'),level=logging.INFO, format=logging_str,
filemode='a')

def main(data,eta, epochs,filename,plotfilename):


    df = pd.DataFrame(OR)
    logging.info(f'this is my actual data: \n{df}')

    X, y = prepare_data(df)

    model  = Perceptron(eta= ETA, epochs= EPOCHS)
    model.fit(X,y)

    _= model.total_loss()

    save_model(model=model,filename=filename)
    save_plot(df,plotfilename,model )

if __name__ =="__main__":
    OR ={
        "x1" : [0,0,1,1],
        "x2" : [0,1,0,1],
        "y" : [0,1,1,1]
    }
    ETA = 0.3
    EPOCHS = 1000
    try:
        logging.info("\n>>>>> starting traing >>>>>")
        main(data=OR, eta=ETA,epochs=EPOCHS,filename='or.model',plotfilename='or.png')
        logging.info("<<<<< stoping the traing >>>>>")
    except Exception as e:
        logging.exception(e)
        raise e