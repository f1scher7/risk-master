from enum import Enum


class Screen(Enum):
    DASHBOARD = "dashboard"
    PRICE_PREDICTION = "price_prediction"
    RISK_ANALYSIS = "risk_analysis"


class Column(Enum):
    TIMESTAMP = "Timestamp"
    OPEN = "Open"
    LOW = "Low"
    HIGH = "High"
    CLOSE = "Close"
    VOLUME = "Volume"


class InvestmentSymbol(Enum):
    BITCOIN = "btc"
    GOLD = "gold"
    SILVER = "silver"
    ETHEREUM = "eth"
    POLSKAGORACOIN = "pgc"
    DOGECOIN = "doge"
