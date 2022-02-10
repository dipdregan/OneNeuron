from fileinput import filename
from utils.model import Perceptron
from utils.all_utils import prepare_data,save_model, save_plot
import pandas as pd
import numpy as np
import joblib

def main(data,eta,epochs,filename,plotfilename):

    df = pd.DataFrame(data)

    X, y = prepare_data(df)



    model  = Perceptron(eta= ETA, epochs= EPOCHS)
    model.fit(X,y)

    _= model.total_loss()

    save_model(model=model,filename=filename)
    save_plot(df, file_name=plotfilename,model=model)

if __name__=="__main":

    AND ={
            "x1" : [0,0,1,1],
            "x2" : [0,1,0,1],
            "y" : [0,0,0,1]
        }
    ETA = 0.3
    EPOCHS = 10

    main(data=AND, eta=ETA,epochs=EPOCHS,filename='and.model',plotfilename='and.png')