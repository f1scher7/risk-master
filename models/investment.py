from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

class Investment(Base):
    __tablename__ = 'investments'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    symbol = Column(String(10), unique=True, nullable=False)
    img_path = Column(String(255), nullable=False)

    _price = None
    _percent_change = None


    @property
    def price(self):
        return self._price

    @property
    def percent_change(self):
        return self._percent_change


    @price.setter
    def price(self, value):
        self._price = value

    @percent_change.setter
    def percent_change(self, value):
        self._percent_change = value


    def __repr__(self):
        return f"Investment(id={self.id}, name={self.name}, symbol={self.symbol}, price={self.price}, percent={self.percent_change})"