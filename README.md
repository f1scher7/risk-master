# RiskMaster

RiskMaster is an intelligent platform designed for individual investors to assess financial and cryptocurrency market risks. It combines historical trend analysis, predictive Fischer AI (LSTM + Dense), and scenario simulations to provide actionable insights for smarter investment decisions.

![image](https://github.com/user-attachments/assets/05e28330-1e72-4993-979b-71b2f8dff01e)

![image](https://github.com/user-attachments/assets/66dbaba0-4124-491e-80dc-e2011d504314)

![image](https://github.com/user-attachments/assets/b392f23d-d453-49f2-ad84-d99331bd2675)


## Local setup

* `pip install -r requirements.txt`
* `docker run -d --name postgres-latest-riskmaster maskfischer7/postgres-latest-riskmaster:1.0`
* Change the img_path column in public.investments table
* Create .env file (check env_loader.py)
* `bash scripts/run-local.sh`


## Datasets

Create folder data_sets in the fischerAI/

* Gold - https://www.kaggle.com/datasets/sid321axn/gold-price-prediction-dataset
* Bitcoin - https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
* Silver - https://www.kaggle.com/datasets/balabaskar/silver-prices-1968-2021
* Ethereum - https://www.kaggle.com/datasets/varpit94/ethereum-data
* DogeCoin -  https://www.kaggle.com/datasets/dhruvildave/dogecoin-historical-data

## Author

Maks Szyło maksymilian.fischer7@gmail.com
