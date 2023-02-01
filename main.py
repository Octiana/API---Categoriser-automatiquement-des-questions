from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from fonctions import *

app_pred_SO = FastAPI(title='API for tags prediction on Stack Overflow posts',
              description='Return tags related to a Stack Overflow poste',
              version='0.0.1')


@app_pred_SO.get("/")
def root():
    return {"Welcome to the API. Check /docs for usage"}


class Input(BaseModel):
    text : str


@app_pred_SO.post("/predict")
async def get_prediction(data: Input):

    text_preprocessed = transform_bow_fct(data.text)
 #  lda_model = LdaModel()
 #   unsupervised_pred = lda_model.predict_tags(text_preprocessed)
    supervised_model = SupervisedModel()
    supervised_pred = supervised_model.predict_tags(text_preprocessed)
    text = jsonable_encoder(data.text)

    return JSONResponse(status_code=200, content={"text": text,
                                                #  "unsupervised_tags": unsupervised_pred,
                                                  "supervised_tags": supervised_pred})
