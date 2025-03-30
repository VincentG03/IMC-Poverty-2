from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json

class Trader:
    def __init__(self):
        # Initialize state
        self.positions = {}
        self.fair_values = {}
        self.position_limits = {
            "STARFRUIT": 20,
            "AMETHYSTS": 20,
            "ORCHIDS": 20,
            "GIFT_BASKET": 5,
            # Add other products as needed
        }
    
    def load_state(self, state_data: str):
        """Load state from previous execution"""
        if state_data and state_data != "SAMPLE":
            try:
                saved_state = json.loads(state_data)
                self.positions = saved_state.get("positions", {})
                self.fair_values = saved_state.get("fair_values", {})
            except:
                pass  # Handle initialization errors
    
    def save_state(self) -> str:
        """Save state for next execution"""
        state = {
            "positions": self.positions,
            "fair_values": self.fair_values
        }
        return json.dumps(state)
    
    def calculate_fair_value(self, product: str, order_depth: OrderDepth) -> float:
        """Calculate dynamic fair value"""
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        return self.fair_values.get(product, 10)  # Fallback
    
    def market_making_strategy(self, product: str, order_depth: OrderDepth, fair_value: float) -> List[Order]:
        """Market making strategy"""
        orders = []
        current_position = self.positions.get(product, 0)
        position_limit = self.position_limits.get(product, 20)
        
        # Dynamic spread and size based on position
        spread = 2 if product == "AMETHYSTS" else 1
        position_ratio = abs(current_position) / position_limit if position_limit > 0 else 0
        base_volume = max(1, int(5 * (1 - position_ratio)))
        
        # Position-based pricing adjustments
        position_bias = -0.1 * current_position
        bid_price = int(fair_value - spread/2 + position_bias)
        ask_price = int(fair_value + spread/2 + position_bias)
        
        # Place orders respecting position limits
        if current_position < position_limit:
            buy_volume = min(position_limit - current_position, base_volume)
            orders.append(Order(product, bid_price, buy_volume))
        
        if current_position > -position_limit:
            sell_volume = -min(position_limit + current_position, base_volume)
            orders.append(Order(product, ask_price, sell_volume))
        
        return orders
    
    def market_taking_strategy(self, product: str, order_depth: OrderDepth, fair_value: float) -> List[Order]:
        """Market taking strategy for mispriced assets"""
        orders = []
        current_position = self.positions.get(product, 0)
        position_limit = self.position_limits.get(product, 20)
        
        # Buy underpriced assets
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = order_depth.sell_orders[best_ask]
            if best_ask < fair_value and current_position < position_limit:
                buy_volume = min(position_limit - current_position, abs(best_ask_volume))
                orders.append(Order(product, best_ask, buy_volume))
        
        # Sell overpriced assets
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]
            if best_bid > fair_value and current_position > -position_limit:
                sell_volume = -min(position_limit + current_position, best_bid_volume)
                orders.append(Order(product, best_bid, sell_volume))
        
        return orders
    
    def basket_trading_strategy(self, product: str, order_depth: OrderDepth, fair_value: float) -> List[Order]:
        """Strategy for basket products like GIFT_BASKET"""
        # Special handling for composite products
        # Logic would depend on specific basket composition
        return self.market_taking_strategy(product, order_depth, fair_value)
    
    def select_strategy(self, product: str) -> str:
        """Select appropriate strategy for each product"""
        if product in ["STARFRUIT", "AMETHYSTS"]:
            return "market_making"
        elif product == "ORCHIDS":
            return "market_taking"
        elif product == "GIFT_BASKET":
            return "basket_trading"
        return "market_making"  # Default
    
    def run(self, state: TradingState):
        # Load previous state
        self.load_state(state.traderData)
        
        # Update positions from current state
        for product, position in state.position.items():
            self.positions[product] = position
        
        print("Current positions:", self.positions)
        print("Observations:", state.observations)
        
        # Process each product
        result = {}
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            
            # Calculate fair value
            fair_value = self.calculate_fair_value(product, order_depth)
            self.fair_values[product] = fair_value
            print(f"Product: {product}, Fair value: {fair_value}")
            
            # Select and execute strategy
            strategy_type = self.select_strategy(product)
            orders = []
            
            if strategy_type == "market_making":
                orders = self.market_making_strategy(product, order_depth, fair_value)
            elif strategy_type == "market_taking":
                orders = self.market_taking_strategy(product, order_depth, fair_value)
            elif strategy_type == "basket_trading":
                orders = self.basket_trading_strategy(product, order_depth, fair_value)
            
            # Log orders for debugging
            for order in orders:
                action = "BUY" if order.quantity > 0 else "SELL"
                print(f"{action} {abs(order.quantity)}x {product} @ {order.price}")
            
            result[product] = orders
        
        # Save state for next execution
        traderData = self.save_state()
        
        # Handle conversions if needed (0 means no conversions)
        conversions = 0
        
        return result, conversions, traderData
