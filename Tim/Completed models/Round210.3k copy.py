import pandas as pd
import json

class VolcanicRockTrader:
    def __init__(self, option_data, current_positions=None):
        """
        option_data: DataFrame with at least 'product' and 'mid_price'
        current_positions: Dict of current position per product
        """
        self.option_data = option_data.copy()
        self.orders = []
        self.current_positions = current_positions if current_positions else {}

    def extract_strike(self):
        self.option_data['strike_price'] = self.option_data['product'].str.extract(r'_(\d+)$').astype(float)

    def classify_and_trade(self):
        def get_action(strike):
            if strike < 10000:
                return 'BUY'
            elif strike == 10500:
                return 'SELL'
            return 'HOLD'

        self.option_data['action'] = self.option_data['strike_price'].apply(get_action)
        self.option_data['quantity'] = self.option_data['action'].apply(lambda x: 200 if x in ['BUY', 'SELL'] else 0)

    def place_orders(self):
        actionable = self.option_data[self.option_data['action'].isin(['BUY', 'SELL'])]

        for _, row in actionable.iterrows():
            product = row['product']
            action = row['action']
            quantity = int(row['quantity'])
            current_pos = self.current_positions.get(product, 0)

            # Enforce position limit of ±200
            if action == 'BUY' and current_pos + quantity > 200:
                continue
            if action == 'SELL' and current_pos - quantity < -200:
                continue

            # Set limit price (slightly better than mid)
            limit_price = row['mid_price'] * (0.99 if action == 'BUY' else 1.01)

            self.orders.append({
                'product': product,
                'action': action,
                'quantity': quantity,
                'mid_price': row['mid_price'],
                'limit_price': round(limit_price, 2)
            })

    def run(self):
        self.extract_strike()
        self.classify_and_trade()
        self.place_orders()
        return self.orders


# Example Lambda-style handler (can also be used locally)
def lambda_handler(event=None, context=None):
    # Example input (replace this with your real data)
    option_data = pd.DataFrame([
        {'product': 'VOLCANIC_ROCK_VOUCHER_9500', 'mid_price': 720},
        {'product': 'VOLCANIC_ROCK_VOUCHER_9700', 'mid_price': 650},
        {'product': 'VOLCANIC_ROCK_VOUCHER_10500', 'mid_price': 500},
        {'product': 'VOLCANIC_ROCK_VOUCHER_10200', 'mid_price': 540},
    ])

    current_positions = {
        'VOLCANIC_ROCK_VOUCHER_9500': 100,
        'VOLCANIC_ROCK_VOUCHER_10500': -200
    }

    trader = VolcanicRockTrader(option_data=option_data, current_positions=current_positions)
    orders = trader.run()

    # For Lambda
    return {
        'statusCode': 200,
        'body': json.dumps(orders, indent=2)
    }


# For local testing (optional)
if __name__ == "__main__":
    result = lambda_handler()
    print(json.dumps(result, indent=2))
