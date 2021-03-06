from utils.model import Perceptron
from utils.all_utils import prepare_data,save_model, save_plot
import pandas as pd
import numpy as np
 
def main():
    XOR ={
        "x1" : [0,0,1,1],
        "x2" : [0,1,0,1],
        "y" : [0,1,1,0]
    }

    df = pd.DataFrame(XOR)

    X, y = prepare_data(df)

    ETA = 0.3
    EPOCHS = 10

    model  = Perceptron(eta= ETA, epochs= EPOCHS)
    model.fit(X,y)

    _= model.total_loss()

    save_model(model=model,filename='XOR.model')
    save_plot(df,'XOR.png',model )

if __name__=="__main__":
    main()