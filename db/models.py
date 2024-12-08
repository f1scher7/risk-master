import datetime
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    session_token = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.now(datetime.UTC))

    def __repr__(self):
        return f"User(id={self.id}, username={self.username}, created_at={self.created_at})"


class Investment(Base):
    __tablename__ = 'investments'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    symbol = Column(String(10), unique=True, nullable=False)
    img_path = Column(String(255), nullable=False)

    def __repr__(self):
        return f"Investment(id={self.id}, name={self.name}, symbol={self.symbol})"