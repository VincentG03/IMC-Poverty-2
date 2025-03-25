import json
from typing import Any
from typing import List, Dict, Tuple
import numpy as np
import statistics
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def __init__(self):
        self.price_history = {}
        self.hold_time = {}

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result = {}
        conversions = 0
        MAX_POSITION = 50

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            current_position = state.position.get(product, 0)

            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

            if product == "KELP":
                if product == "KELP":
                    if best_bid is not None and best_ask is not None:
                        spread = best_ask - best_bid
                        min_spread = 2

                        spread_factor = max(1, spread)
                        inventory_factor = max(0.2, 1 - abs(current_position) / MAX_POSITION)
                        base_size = 6
                        max_size = 20

                        order_size = int(base_size * spread_factor * inventory_factor)
                        order_size = min(max(order_size, 1), max_size)

                        edge_buffer = 1 if spread > 2 else 0

                        if spread >= min_spread:
                            if current_position > -MAX_POSITION:
                                orders.append(Order(product, best_bid + edge_buffer, order_size))
                            if current_position < MAX_POSITION:
                                orders.append(Order(product, best_ask - edge_buffer, -order_size))
                        elif abs(current_position) > 0:
                            fair_price = (best_bid + best_ask) // 2
                            rebalance_size = min(order_size, abs(current_position))
                            if current_position > 0:
                                orders.append(Order(product, fair_price, -rebalance_size))
                            else:
                                orders.append(Order(product, fair_price, rebalance_size))



                
            elif product == "RAINFOREST_RESIN":
                if best_bid is not None and best_ask is not None:
                    mid_price = (best_bid + best_ask) / 2

                    if product not in self.price_history:
                        self.price_history[product] = []
                    
                    self.price_history[product].append(mid_price)
                    if len(self.price_history[product]) >10:
                        self.price_history[product].pop(0)
                            
                    if len(self.price_history[product]) >=10:
                        mean_price = statistics.mean(self.price_history[product])
                        std_price = statistics.stdev(self.price_history[product])
                        z_score = (mid_price - mean_price) / std_price if std_price > 0 else 0

                        order_size = 20

                        if z_score > 0.1 and current_position - order_size >= -MAX_POSITION:
                            orders.append(Order(product, best_ask, - order_size))
                        elif z_score < -0.1 and current_position + order_size <= MAX_POSITION:
                            orders.append(Order(product, best_bid, order_size))
                        elif abs(z_score) < 0.1:
                            if current_position > 0:
                                orders.append(Order(product, best_ask, -current_position))
                            elif current_position < 0:
                                orders.append(Order(product, best_bid, -current_position))

            else:
                logger.print("Unknown product:", product)

            result[product] = orders

        traderData = "GRID+MOM"
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData