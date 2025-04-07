from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json
import time

class Trader:
    def __init__(self):
        # Initialize state
        self.positions = {}
        self.fair_values = {}
        self.position_limits = {
            "RAINFOREST_RESIN": 20,
            "KELP": 15
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
        """Hybrid fair value model combining order book imbalance and ecological factors"""
        if product == "RAINFOREST_RESIN":
            # Stable commodity pricing model
            return (max(order_depth.buy_orders.keys()) + min(order_depth.sell_orders.keys())) / 2
        else:
            # Kelp momentum-adjusted pricing
            vwap = sum(p*q for p,q in order_depth.buy_orders.items() + order_depth.sell_orders.items()) / sum(order_depth.buy_orders.values() + order_depth.sell_orders.values())
            momentum = self.calculate_momentum(product)
            return vwap * (1 + 0.15 * momentum)  # 15% momentum factor
    
    def calculate_momentum(self, product: str) -> float:
        """Computer vision-inspired trend detection from order flow"""
        if product != "KELP":
            return 0.0
        
        # Analyze order book using algal growth models
        price_changes = []
        for _ in range(5):
            current_best_bid_ask = self.get_best_bid_ask(product)
            time.sleep(0.1)
            new_best_bid_ask = self.get_best_bid_ask(product)
            price_changes.append(new_best_bid_ask[0] - current_best_bid_ask[0])
        
        # Biomass estimation techniques applied to price movement
        return sum(pc > 0 for pc in price_changes)/5 - 0.5
    
    def get_best_bid_ask(self, product: str) -> tuple:
        """Helper function to get best bid and ask prices"""
        order_depth = self.order_depths.get(product, {})
        best_bid = max(order_depth.buy_orders.keys(), default=0)
        best_ask = min(order_depth.sell_orders.keys(), default=float('inf'))
        return best_bid, best_ask
    
    def resin_strategy(self, order_depth: OrderDepth, position: int) -> List[Order]:
        """Mean-reversion market making strategy for Rainforest Resin"""
        orders = []
        position_limit = self.position_limits["RAINFOREST_RESIN"]
        spread = 1  # Minimum tick size
        
        # Dynamic inventory adjustment
        inventory_penalty = -0.05 * position
        bid_price = self.fair_values["RAINFOREST_RESIN"] - spread/2 + inventory_penalty
        ask_price = self.fair_values["RAINFOREST_RESIN"] + spread/2 + inventory_penalty
        
        # Order sizing based on position limits
        base_size = min(5, position_limit - abs(position))
        
        if position < position_limit:
            orders.append(Order("RAINFOREST_RESIN", bid_price, base_size))
        if position > -position_limit:
            orders.append(Order("RAINFOREST_RESIN", ask_price, -base_size))
        
        return orders
    
    def kelp_strategy(self, order_depth: OrderDepth, position: int) -> List[Order]:
        """Momentum-enhanced pendulum strategy for Kelp"""
        orders = []
        position_limit = self.position_limits["KELP"]
        momentum = self.calculate_momentum("KELP")
        
        # Dynamic spread based on momentum
        spread = max(3, int(5 * (1 - abs(momentum))))
        
        if momentum > 0.2:
            # Bull kelp phase - aggressive buying
            bid_price = self.fair_values["KELP"] - spread
            ask_price = self.fair_values["KELP"] + spread * 3
        elif momentum < -0.2:
            # Kelp forest decline phase - conservative pricing
            bid_price = self.fair_values["KELP"] - spread * 3
            ask_price = self.fair_values["KELP"] + spread
        else:
            # Stable growth period
            bid_price = self.fair_values["KELP"] - spread
            ask_price = self.fair_values["KELP"] + spread
        
        # Order sizes follow position limits
        max_size = min(8, position_limit - abs(position))
        
        if position < position_limit:
            orders.append(Order("KELP", bid_price, max_size))
        
        if position > -position_limit:
            orders.append(Order("KELP", ask_price, -max_size))
        
        return orders
    
    def risk_check(self, product: str, proposed_orders: List[Order]) -> List[Order]:
        """Implements risk controls based on position limits"""
        current_position = self.positions.get(product, 0)
        safe_orders = []
        
        for order in proposed_orders:
            projected_position = current_position + order.quantity
            position_limit = self.position_limits[product]
            
            if abs(projected_position) <= position_limit:
                safe_orders.append(order)
        
        return safe_orders
    
    def run(self, state: TradingState):
        """Main execution method"""
        
        # Load previous state data
        self.load_state(state.traderData)
        
        # Update positions and fair values from current state
        for product in state.position.keys():
            self.positions[product] = state.position[product]
        
        result = {}
        
        for product in state.order_depths.keys():
            order_depth: OrderDepth = state.order_depths[product]
            
            # Calculate fair value dynamically
            fair_value = self.calculate_fair_value(product, order_depth)
            self.fair_values[product] = fair_value
            
            print(f"Product: {product}, Fair Value: {fair_value}")
            
            # Select and execute strategy based on product type
            if product == "RAINFOREST_RESIN":
                proposed_orders = self.resin_strategy(order_depth, self.positions.get(product, 0))
            
            elif product == "KELP":
                proposed_orders = self.kelp_strategy(order_depth, self.positions.get(product, 0))
            
            else:
                proposed_orders = []
            
            # Apply risk checks to proposed orders
            safe_orders = self.risk_check(product, proposed_orders)
            
            result[product] = safe_orders
            
            for order in safe_orders:
                action_type = "BUY" if order.quantity > 0 else "SELL"
                print(f"{action_type} {abs(order.quantity)}x {product} @ {order.price}")
        
        traderData = self.save_state()
        
        conversions_needed = 0
        
        return result, conversions_needed, traderData
