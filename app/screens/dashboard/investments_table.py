import time
import ccxt
from services.investment_service import InvestmentService
from utils.app_utils import *
from utils.prices_fetcher_utils import fetch_all_prices


investment_symbols = ['GOLD', 'SILVER', 'BTC', 'ETH', 'PGC', 'DOGE']
investments = [InvestmentService.get_investment_by_symbol(symbol) for symbol in investment_symbols]


def update_investments_table(table_placeholder):
    exchange = ccxt.binance()

    previous_prices = fetch_all_prices(exchange)

    try:
        while True:
            current_prices = fetch_all_prices(exchange)

            percent_changes = {
                investment: round(((current_prices[investment] - previous_prices[investment]) / previous_prices[investment] * 100), 2)
                for investment in current_prices
            }

            for investment in investments:
                investment.price = current_prices[investment.name]
                investment.percent_change = percent_changes[investment.name]

            previous_prices = current_prices

            table_placeholder.markdown(generate_investments_table(), unsafe_allow_html=True)

            time.sleep(3)
    except Exception as e:
        print(f"Error occurred: {e}")


def generate_investments_table():
    load_css("investmentsTable.css")

    investments_info = [InvestmentService.get_investment_info_text(investment) for investment in investments]


    return f"""
        <div class="investmentTable">
            <div class="row">
                <div class="tableCell">
                    <img class="logoImg" src="data:image/png;base64,{encode_image_to_base64(investments[0].img_path)}">
                    <div class="investmentInfo">
                        <h4 class="logoName">{investments[0].name}</h4>
                        <p class="price">{investments_info[0]}</p>
                    </div>
                </div>
                <div class="tableCell">
                    <img class="logoImg" src="data:image/png;base64,{encode_image_to_base64(investments[1].img_path)}">
                    <div class="investmentInfo">
                        <h4 class="logoName">{investments[1].name}</h4>
                        <p class="price">{investments_info[1]}</p>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="tableCell">
                    <img class="logoImg" src="data:image/png;base64,{encode_image_to_base64(investments[2].img_path)}">
                    <div class="investmentInfo">
                        <h4 class="logoName">{investments[2].name}</h4>
                        <p class="price">{investments_info[2]}</p>
                    </div>
                </div>
                <div class="tableCell">
                    <img class="logoImg" src="data:image/png;base64,{encode_image_to_base64(investments[3].img_path)}">
                    <div class="investmentInfo">
                        <h4 class="logoName">{investments[3].name}</h4>
                        <p class="price">{investments_info[3]}</p>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="tableCell">
                    <img class="logoImg" src="data:image/png;base64,{encode_image_to_base64(investments[4].img_path)}">
                    <div class="investmentInfo">
                        <h4 class="logoName">{investments[4].name}</h4>
                        <p class="price">{investments_info[4]}</p>
                    </div>
                </div>
                <div class="tableCell">
                    <img class="logoImg" src="data:image/png;base64,{encode_image_to_base64(investments[5].img_path)}">
                    <div class="investmentInfo">
                        <h4 class="logoName">{investments[5].name}</h4>
                        <p class="price">{investments_info[5]}</p>
                    </div>
                </div>
            </div>
        </div>
    """
