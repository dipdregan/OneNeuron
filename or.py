from utils.model import Perceptron
from utils.all_utils import prepare_data,save_model, save_plot
import pandas as pd
import numpy as np
import joblib
import logging

logging_str = "[%(asctime)s:%(module)s:%(levelname)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=logging_str)

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
    EPOCHS = 10
    try:
        logging.info(">>>>> starting traing >>>>>")
        main(data=OR, eta=ETA,epochs=EPOCHS,filename='or.model',plotfilename='or.png')
        logging.info("<<<<< stoping the traing >>>>>")
    except Exception as e:
        logging.exception(e)