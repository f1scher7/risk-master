from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

class Investment(Base):
    __tablename__ = 'investments'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    symbol = Column(String(10), unique=True, nullable=False)
    img_path = Column(String(255), nullable=False)

    def __repr__(self):
        return f"Investment(id={self.id}, name={self.name}, symbol={self.symbol})"