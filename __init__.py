import Projet5.API
from Projet5.API import main
from main import app_pred_SO


# Creation API dans un host serveur sur l'ordi seulement
import uvicorn

if __name__ == "main":
   uvicorn.run(app_pred_SO, host='127.0.0.1', port=5000, log_level="info")


   #http://127.0.0.1:5000/
    # uvicorn --port 5000 --host 127.0.0.1 main:app_pred_SO --reload
