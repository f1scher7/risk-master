from db.models import Investment
from db.db_connection import get_session
from sqlalchemy.exc import IntegrityError


class InvestmentService:
    @staticmethod
    def create_investment(name: str, symbol: str):
        try:
            investment = Investment(name=name, symbol=symbol)

            with get_session() as db_session:
                db_session.add(investment)
                db_session.commit()

            return investment
        except IntegrityError:
            return {"ERROR": f"Investment with symbol '{symbol}' already exists"}

    @staticmethod
    def get_investment_by_id(investment_id: int):
        with get_session() as db_session:
            investment = db_session.query(Investment).filter_by(id=investment_id).first()
        return investment

    @staticmethod
    def get_investment_by_symbol(symbol: str):
        with get_session() as db_session:
            investment = db_session.query(Investment).filter_by(symbol=symbol).first()
        return investment

    @staticmethod
    def get_all_investments():
        with get_session() as db_session:
            investments = db_session.query(Investment).all()
        return investments

    @staticmethod
    def delete_investment_by_symbol(symbol: str):
        investment = InvestmentService.get_investment_by_symbol(symbol)

        if investment:
            with get_session() as db_session:
                db_session.delete(investment)
                db_session.commit()
            return {"SUCCESS": f"Investment with symbol '{symbol}' was deleted"}
        else:
            return {"ERROR": f"Investment with symbol '{symbol}' not found"}
