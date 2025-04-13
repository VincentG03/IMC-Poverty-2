
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
import json
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
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""

        # TODO: Add logic

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
    
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN" #RESIN, FROM A 
    KELP = "KELP" #KELP, FROM S
    # ORCHIDS = "ORCHIDS" #WE DONT FUCK ORCHIDS
    PICNIC_BASKET1 = "PICNIC_BASKET1" #GIFT BASKET 1 
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS" #CROISSANTS FROM C
    JAMS = "JAMS" #JAMS FROM S
    DJEMBES = "DJEMBES" #DJEMBES FROM R
    SYNTHETIC1 = "SYNTHETIC1"
    SYNTHETIC2 ="SYNTHETIC2"
    SPREAD1 = "SPREAD1"
    SPREAD2 = "SPREAD2"
    SQUID_INK = "SQUID_INK"
    

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0.5,
        "volume_limit": 0,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "kelp_min_edge": 2,
    },
    Product.SQUID_INK: {
    "take_width": 1,
    "clear_width": 2,
    "make_width": 2.65,
    "adverse_volume": 15,
    "timespan": 10,
    },
    Product.CROISSANTS: {
        "take_width": 1,
        "clear_width": 2,
        "make_width": 2.75,
        "adverse_volume": 15,
        "timespan": 10,
    },
    # Product.ORCHIDS: {
    #     "make_edge": 2,
    #     "make_probability": 0.800,
    # },
    Product.SPREAD1: {
        "default_spread_mean": 48.76243333, #379.50439988484239,
        "default_spread_std": 85.11945081,#76.07966,
        "spread_std_window": 10,
        "zscore_threshold": 2,
        "target_position": 60,
    },
    Product.SPREAD2: {
        "default_spread_mean": 30.23596667, #379.50439988484239,
        "default_spread_std": 59.84920022,#76.07966,
        "spread_std_window": 400,
        "zscore_threshold": 2,
        "target_position": 100,
    },
}

BASKET_WEIGHTS1 = {
    Product.CROISSANTS: 6,#8.42, #This was 4
    Product.JAMS: 3,#3.1, #This was 6
    Product.DJEMBES: 1,#-0.51, #This was 1
}
BASKET_WEIGHTS2 = {
    Product.CROISSANTS: 4,#8.42, #This was 4
    Product.JAMS: 2,#3.1, #This was 6
}




class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.kelp_prices = []
        self.kelp_vwap = []
        self.kelp_mmmid = []
        self.squid_ink_prices = []
        self.squid_ink_vwap = []
        self.squid_ink_mmmid = []

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            # Product.ORCHIDS: 100,
            Product.PICNIC_BASKET1: 60, #PICNICBASKET1
            Product.PICNIC_BASKET2: 100, 
            Product.CROISSANTS: 250, #CROISSANTSS
            Product.JAMS: 350, #JAMS
            Product.DJEMBES: 60, #DJEMBES 
            Product.SQUID_INK: 50
            

        }

    # Returns buy_order_volume, sell_order_volume
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if best_ask <= fair_value - take_width:
                quantity = min(
                    best_ask_amount, position_limit - position
                )  # max amt to buy
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(
                    best_bid_amount, position_limit + position
                )  # should be the max we can sell
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def take_best_orders_with_adverse(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        adverse_volume: int,
    ) -> (int, int):

        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def make_rainforest_resin_orders(
        self,
        order_depth: OrderDepth,
        fair_value: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        volume_limit: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        baaf = min(
            [
                price
                for price in order_depth.sell_orders.keys()
                if price > fair_value + 1
            ]
        )
        bbbf = max(
            [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        )

        if baaf <= fair_value + 2:
            if position <= volume_limit:
                baaf = fair_value + 3  # still want edge 2 if position is not a concern

        if bbbf >= fair_value - 2:
            if position >= -volume_limit:
                bbbf = fair_value - 3  # still want edge 2 if position is not a concern

        buy_order_volume, sell_order_volume = self.market_make(
            Product.RAINFOREST_RESIN,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if prevent_adverse:
            buy_order_volume, sell_order_volume = self.take_best_orders_with_adverse(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
                adverse_volume,
            )
        else:
            buy_order_volume, sell_order_volume = self.take_best_orders(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
            )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_kelp_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        min_edge: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        aaf = [
            price
            for price in order_depth.sell_orders.keys()
            if price >= round(fair_value + min_edge)
        ]
        bbf = [
            price
            for price in order_depth.buy_orders.keys()
            if price <= round(fair_value - min_edge)
        ]
        baaf = min(aaf) if len(aaf) > 0 else round(fair_value + min_edge)
        bbbf = max(bbf) if len(bbf) > 0 else round(fair_value - min_edge)
        buy_order_volume, sell_order_volume = self.market_make(
            Product.KELP,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume
    
    # def jams_orders(self, order_depth: OrderDepth, position: int, timemspan: int = 10) -> List[Order]:
    #     orders: List[Order] = []
    #     buy_order_volume = 0
    #     sell_order_volume = 0
    #     position_limit = self.LIMIT[Product.JAMS]

    #     if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
    #         return orders  # No liquidity

    #     best_bid = max(order_depth.buy_orders.keys())
    #     best_ask = min(order_depth.sell_orders.keys())

    #     filtered_bid = [p for p in order_depth.buy_orders if abs(order_depth.buy_orders[p]) >= 15]
    #     filtered_ask = [p for p in order_depth.sell_orders if abs(order_depth.sell_orders[p]) >= 15]

    #     mm_bid = max(filtered_bid) if filtered_bid else best_bid
    #     mm_ask = min(filtered_ask) if filtered_ask else best_ask

    #     mid = (mm_bid + mm_ask) / 2
    #     fair_value = mid

    #     # Take orders if profitable
    #     if best_ask < fair_value - 1 and position < position_limit:
    #         quantity = min(-order_depth.sell_orders[best_ask], position_limit - position)
    #         orders.append(Order(Product.JAMS, best_ask, quantity))

    #     if best_bid > fair_value + 1 and position > -position_limit:
    #         quantity = min(order_depth.buy_orders[best_bid], position_limit + position)
    #         orders.append(Order(Product.JAMS, best_bid, -quantity))

    #     # Market make around fair value
    #     spread = 3  # Customize this
    #     buy_price = round(fair_value - spread)
    #     sell_price = round(fair_value + spread)

    #     if position + buy_order_volume < position_limit:
    #         orders.append(Order(Product.JAMS, buy_price, position_limit - position))

    #     if position - sell_order_volume > -position_limit:
    #         orders.append(Order(Product.JAMS, sell_price, -position_limit - position))

    #     return orders
    
    
    # def croissants_orders(self, order_depth: OrderDepth, position: int,) -> List[Order]:
    #     orders: List[Order] = []

    #     buy_order_volume = 0
    #     sell_order_volume = 0
    #     params = self.params[Product.CROISSANTS]
    #     timespan = params["timespan"]

    #     if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
    #         best_ask = min(order_depth.sell_orders.keys())
    #         best_bid = max(order_depth.buy_orders.keys())

    #         filtered_ask = [price for price in order_depth.sell_orders if abs(order_depth.sell_orders[price]) >= params["adverse_volume"]]
    #         filtered_bid = [price for price in order_depth.buy_orders if abs(order_depth.buy_orders[price]) >= params["adverse_volume"]]

    #         mm_ask = min(filtered_ask) if filtered_ask else best_ask
    #         mm_bid = max(filtered_bid) if filtered_bid else best_bid
    #         mmmid_price = (mm_ask + mm_bid) / 2

    #         # Fair value history tracking (optional, you can skip if not used)
    #         self.squid_ink_prices.append(mmmid_price)
    #         if len(self.squid_ink_prices) > timespan:
    #             self.squid_ink_prices.pop(0)

    #         fair_value = mmmid_price

    #         # Take best orders
    #         buy_order_volume, sell_order_volume = self.take_best_orders(
    #             Product.CROISSANTS,
    #             fair_value,
    #             params["take_width"],
    #             orders,
    #             order_depth,
    #             position,
    #             buy_order_volume,
    #             sell_order_volume,
    #             True,
    #             params["adverse_volume"]
    #         )

    #         # Clear orders
    #         buy_order_volume, sell_order_volume = self.clear_position_order(
    #             Product.CROISSANTS,
    #             fair_value,
    #             params["clear_width"],
    #             orders,
    #             order_depth,
    #             position,
    #             buy_order_volume,
    #             sell_order_volume
    #         )

    #         # Market making
    #         aaf = [p for p in order_depth.sell_orders if p > fair_value + 1]
    #         bbf = [p for p in order_depth.buy_orders if p < fair_value - 1]
    #         baaf = min(aaf) if aaf else fair_value + 2
    #         bbbf = max(bbf) if bbf else fair_value - 2

    #         buy_order_volume, sell_order_volume = self.market_make(
    #             Product.CROISSANTS,
    #             orders,
    #             bbbf + 1,
    #             baaf - 1,
    #             position,
    #             buy_order_volume,
    #             sell_order_volume
    #         )

    #     return orders



    # def orchids_implied_bid_ask(
    #     self,
    #     observation: ConversionObservation,
    # ) -> (float, float):
    #     return (
    #         observation.bidPrice
    #         - observation.exportTariff
    #         - observation.transportFees
    #         - 0.1,
    #         observation.askPrice + observation.importTariff + observation.transportFees,
    #     )

    # def orchids_arb_take(
    #     self,
    #     order_depth: OrderDepth,
    #     observation: ConversionObservation,
    #     position: int,
    # ) -> (List[Order], int, int):
    #     orders: List[Order] = []
    #     position_limit = self.LIMIT[Product.ORCHIDS]
    #     buy_order_volume = 0
    #     sell_order_volume = 0

    #     implied_bid, implied_ask = self.orchids_implied_bid_ask(observation)

    #     buy_quantity = position_limit - position
    #     sell_quantity = position_limit + position

    #     ask = round(observation.askPrice) - 2

    #     if ask > implied_ask:
    #         edge = (ask - implied_ask) * self.params[Product.ORCHIDS]["make_probability"]
    #     else:
    #         edge = 0

    #     for price in sorted(list(order_depth.sell_orders.keys())):
    #         if price > implied_bid - edge:
    #             break

    #         if price < implied_bid - edge:
    #             quantity = min(
    #                 abs(order_depth.sell_orders[price]), buy_quantity
    #             )  # max amount to buy
    #             if quantity > 0:
    #                 orders.append(Order(Product.ORCHIDS, round(price), quantity))
    #                 buy_order_volume += quantity

    #     for price in sorted(list(order_depth.buy_orders.keys()), reverse=True):
    #         if price < implied_ask + edge:
    #             break

    #         if price > implied_ask + edge:
    #             quantity = min(
    #                 abs(order_depth.buy_orders[price]), sell_quantity
    #             )  # max amount to sell
    #             if quantity > 0:
    #                 orders.append(Order(Product.ORCHIDS, round(price), -quantity))
    #                 sell_order_volume += quantity

    #     return orders, buy_order_volume, sell_order_volume

    # def orchids_arb_clear(self, position: int) -> int:
    #     conversions = -position
    #     return conversions

    # def orchids_arb_make(
    #     self,
    #     observation: ConversionObservation,
    #     position: int,
    #     buy_order_volume: int,
    #     sell_order_volume: int,
    # ) -> (List[Order], int, int):
    #     orders: List[Order] = []
    #     position_limit = self.LIMIT[Product.ORCHIDS]

    #     # Implied Bid = observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1
    #     # Implied Ask = observation.askPrice + observation.importTariff + observation.transportFees
    #     implied_bid, implied_ask = self.orchids_implied_bid_ask(observation)

    #     aggressive_ask = round(observation.askPrice) - 2
    #     aggressive_bid = round(observation.bidPrice) + 2

    #     if aggressive_bid < implied_bid:
    #         bid = aggressive_bid
    #     else:
    #         bid = implied_bid - 1

    #     if aggressive_ask >= implied_ask + 0.5:
    #         ask = aggressive_ask
    #     elif aggressive_ask + 1 >= implied_ask + 0.5:
    #         ask = aggressive_ask + 1
    #     else:
    #         ask = implied_ask + 2

    #     print(f"ALGO_ASK: {round(ask)}")
    #     print(f"IMPLIED_BID: {implied_bid}")
    #     print(f"IMPLIED_ASK: {implied_ask}")
    #     print(f"FOREIGN_ASK: {observation.askPrice}")
    #     print(f"FOREIGN_BID: {observation.bidPrice}")

    #     buy_quantity = position_limit - (position + buy_order_volume)
    #     if buy_quantity > 0:
    #         orders.append(Order(Product.ORCHIDS, round(bid), buy_quantity))

    #     sell_quantity = position_limit + (position - sell_order_volume)
    #     if sell_quantity > 0:
    #         orders.append(Order(Product.ORCHIDS, round(ask), -sell_quantity))

    #     return orders, buy_order_volume, sell_order_volume

    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def get_synthetic1_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        CROISSANTS_PER_BASKET1 = BASKET_WEIGHTS1[Product.CROISSANTS]
        JAMS_PER_BASKET1 = BASKET_WEIGHTS1[Product.JAMS]
        DJEMBES_PER_BASKET1 = BASKET_WEIGHTS1[Product.DJEMBES]

        # Initialize the synthetic basket order depth
        synthetic1_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        croissants_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        croissants_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        jams_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        jams_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )
        djembes_best_bid = (
            max(order_depths[Product.DJEMBES].buy_orders.keys())
            if order_depths[Product.DJEMBES].buy_orders
            else 0
        )
        djembes_best_ask = (
            min(order_depths[Product.DJEMBES].sell_orders.keys())
            if order_depths[Product.DJEMBES].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
            croissants_best_bid * CROISSANTS_PER_BASKET1
            + jams_best_bid * JAMS_PER_BASKET1
            + djembes_best_bid * DJEMBES_PER_BASKET1
        )
        implied_ask = (
            croissants_best_ask * CROISSANTS_PER_BASKET1
            + jams_best_ask * JAMS_PER_BASKET1
            + djembes_best_ask * DJEMBES_PER_BASKET1
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            croissants_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[croissants_best_bid]
                // CROISSANTS_PER_BASKET1
            )
            jams_bid_volume = (
                order_depths[Product.JAMS].buy_orders[jams_best_bid]
                // JAMS_PER_BASKET1
            )
            djembes_bid_volume = (
                order_depths[Product.DJEMBES].buy_orders[djembes_best_bid]
                // DJEMBES_PER_BASKET1
            )
            implied_bid_volume = min(
                croissants_bid_volume, jams_bid_volume, djembes_bid_volume
            )
            synthetic1_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            croissants_ask_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[croissants_best_ask]
                // CROISSANTS_PER_BASKET1
            )
            jams_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[jams_best_ask]
                // JAMS_PER_BASKET1
            )
            djembes_ask_volume = (
                -order_depths[Product.DJEMBES].sell_orders[djembes_best_ask]
                // DJEMBES_PER_BASKET1
            )
            implied_ask_volume = min(
                croissants_ask_volume, jams_ask_volume, djembes_ask_volume
            )
            synthetic1_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic1_order_price

    def convert_synthetic1_basket_orders(
        self, synthetic1_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
            Product.DJEMBES: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic1_basket_order_depth = self.get_synthetic1_basket_order_depth(
            order_depths
        )
        best_bid = (
            max(synthetic1_basket_order_depth.buy_orders.keys())
            if synthetic1_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic1_basket_order_depth.sell_orders.keys())
            if synthetic1_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic1_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                croissants_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                jams_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
                djembes_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                croissants_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                jams_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
                djembes_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            croissants_order = Order(
                Product.CROISSANTS,
                croissants_price,
                quantity * BASKET_WEIGHTS1[Product.CROISSANTS],
            )
            jams_order = Order(
                Product.JAMS,
                jams_price,
                quantity * BASKET_WEIGHTS1[Product.JAMS],
            )
            djembes_order = Order(
                Product.DJEMBES, djembes_price, quantity * BASKET_WEIGHTS1[Product.DJEMBES]
            )

            # Add the component orders to the respective lists
            component_orders[Product.CROISSANTS].append(croissants_order)
            component_orders[Product.JAMS].append(jams_order)
            component_orders[Product.DJEMBES].append(djembes_order)

        return component_orders

    def execute_spread1_orders(
        self,
        target1_position: int,
        basket1_position: int,
        order_depths: Dict[str, OrderDepth],
    ):

        if target1_position == basket1_position:
            return None

        target1_quantity = abs(target1_position - basket1_position)
        basket1_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic1_order_depth = self.get_synthetic1_basket_order_depth(order_depths)

        if target1_position > basket1_position:
            basket1_ask_price = min(basket1_order_depth.sell_orders.keys())
            basket1_ask_volume = abs(basket1_order_depth.sell_orders[basket1_ask_price])

            synthetic1_bid_price = max(synthetic1_order_depth.buy_orders.keys())
            synthetic1_bid_volume = abs(
                synthetic1_order_depth.buy_orders[synthetic1_bid_price]
            )

            orderbook_volume = min(basket1_ask_volume, synthetic1_bid_volume)
            execute_volume = min(orderbook_volume, target1_quantity)

            basket1_orders = [
                Order(Product.PICNIC_BASKET1, basket1_ask_price, execute_volume)
            ]
            synthetic1_orders = [
                Order(Product.SYNTHETIC1, synthetic1_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic1_basket_orders(
                synthetic1_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET1] = basket1_orders
            return aggregate_orders

        else:
            basket1_bid_price = max(basket1_order_depth.buy_orders.keys())
            basket1_bid_volume = abs(basket1_order_depth.buy_orders[basket1_bid_price])

            synthetic1_ask_price = min(synthetic1_order_depth.sell_orders.keys())
            synthetic1_ask_volume = abs(
                synthetic1_order_depth.sell_orders[synthetic1_ask_price]
            )

            orderbook_volume = min(basket1_bid_volume, synthetic1_ask_volume)
            execute_volume = min(orderbook_volume, target1_quantity)

            basket1_orders = [
                Order(Product.PICNIC_BASKET1, basket1_bid_price, -execute_volume)
            ]
            synthetic1_orders = [
                Order(Product.SYNTHETIC1, synthetic1_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic1_basket_orders(
                synthetic1_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET1] = basket1_orders
            return aggregate_orders

    def spread1_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket1_position: int,
        spread_data: Dict[str, Any],
    ):
        if Product.PICNIC_BASKET1 not in order_depths.keys():
            return None

        basket1_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic1_order_depth = self.get_synthetic1_basket_order_depth(order_depths)
        basket1_swmid = self.get_swmid(basket1_order_depth)
        synthetic1_swmid = self.get_swmid(synthetic1_order_depth)
        spread1 = basket1_swmid - synthetic1_swmid
        spread_data.setdefault("spread_history").append(spread1)

        if (
            len(spread_data["spread_history"])
            < self.params[Product.SPREAD1]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD1]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread1_std = np.std(spread_data["spread_history"])

        zscore = (
            spread1 - self.params[Product.SPREAD1]["default_spread_mean"]
        ) / spread1_std

        if zscore >= self.params[Product.SPREAD1]["zscore_threshold"]:
            if basket1_position != -self.params[Product.SPREAD1]["target_position"]:
                return self.execute_spread1_orders(
                    -self.params[Product.SPREAD1]["target_position"],
                    basket1_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD1]["zscore_threshold"]:
            if basket1_position != self.params[Product.SPREAD1]["target_position"]:
                return self.execute_spread1_orders(
                    self.params[Product.SPREAD1]["target_position"],
                    basket1_position,
                    order_depths,
                )

        spread_data["prev_zscore"] = zscore
        return None
    
    def get_synthetic2_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        CROISSANTS_PER_BASKET2 = BASKET_WEIGHTS2[Product.CROISSANTS]
        JAMS_PER_BASKET2 = BASKET_WEIGHTS2[Product.JAMS]

        # Initialize the synthetic basket order depth
        synthetic2_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        croissants_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        croissants_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        jams_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        jams_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
            croissants_best_bid * CROISSANTS_PER_BASKET2
            + jams_best_bid * JAMS_PER_BASKET2
          
        )
        implied_ask = (
            croissants_best_ask * CROISSANTS_PER_BASKET2
            + jams_best_ask * JAMS_PER_BASKET2
           
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            croissants_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[croissants_best_bid]
                // CROISSANTS_PER_BASKET2
            )
            jams_bid_volume = (
                order_depths[Product.JAMS].buy_orders[jams_best_bid]
                // JAMS_PER_BASKET2
            )

            implied_bid_volume = min(
                croissants_bid_volume, jams_bid_volume
            )
            synthetic2_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            croissants_ask_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[croissants_best_ask]
                // CROISSANTS_PER_BASKET2
            )
            jams_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[jams_best_ask]
                // JAMS_PER_BASKET2
            )

            implied_ask_volume = min(
                croissants_ask_volume, jams_ask_volume
            )
            synthetic2_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic2_order_price

    def convert_synthetic2_basket_orders(
        self, synthetic2_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic2_basket_order_depth = self.get_synthetic2_basket_order_depth(
            order_depths
        )
        best_bid = (
            max(synthetic2_basket_order_depth.buy_orders.keys())
            if synthetic2_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic2_basket_order_depth.sell_orders.keys())
            if synthetic2_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic2_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                croissants_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                jams_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )

            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                croissants_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                jams_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )

            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            croissants_order = Order(
                Product.CROISSANTS,
                croissants_price,
                quantity * BASKET_WEIGHTS2[Product.CROISSANTS],
            )
            jams_order = Order(
                Product.JAMS,
                jams_price,
                quantity * BASKET_WEIGHTS2[Product.JAMS],
            )


            # Add the component orders to the respective lists
            component_orders[Product.CROISSANTS].append(croissants_order)
            component_orders[Product.JAMS].append(jams_order)


        return component_orders

    def execute_spread2_orders(
        self,
        target2_position: int,
        basket2_position: int,
        order_depths: Dict[str, OrderDepth],
    ):

        if target2_position == basket2_position:
            return None

        target2_quantity = abs(target2_position - basket2_position)
        basket2_order_depth = order_depths[Product.PICNIC_BASKET2]
        synthetic2_order_depth = self.get_synthetic2_basket_order_depth(order_depths)

        if target2_position > basket2_position:
            basket2_ask_price = min(basket2_order_depth.sell_orders.keys())
            basket2_ask_volume = abs(basket2_order_depth.sell_orders[basket2_ask_price])

            synthetic2_bid_price = max(synthetic2_order_depth.buy_orders.keys())
            synthetic2_bid_volume = abs(
                synthetic2_order_depth.buy_orders[synthetic2_bid_price]
            )

            orderbook_volume = min(basket2_ask_volume, synthetic2_bid_volume)
            execute_volume = min(orderbook_volume, target2_quantity)

            basket2_orders = [
                Order(Product.PICNIC_BASKET2, basket2_ask_price, execute_volume)
            ]
            synthetic2_orders = [
                Order(Product.SYNTHETIC2, synthetic2_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic2_basket_orders(
                synthetic2_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET2] = basket2_orders
            return aggregate_orders

        else:
            basket2_bid_price = max(basket2_order_depth.buy_orders.keys())
            basket2_bid_volume = abs(basket2_order_depth.buy_orders[basket2_bid_price])

            synthetic2_ask_price = min(synthetic2_order_depth.sell_orders.keys())
            synthetic2_ask_volume = abs(
                synthetic2_order_depth.sell_orders[synthetic2_ask_price]
            )

            orderbook_volume = min(basket2_bid_volume, synthetic2_ask_volume)
            execute_volume = min(orderbook_volume, target2_quantity)

            basket2_orders = [
                Order(Product.PICNIC_BASKET2, basket2_bid_price, -execute_volume)
            ]
            synthetic2_orders = [
                Order(Product.SYNTHETIC2, synthetic2_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic2_basket_orders(
                synthetic2_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET2] = basket2_orders
            return aggregate_orders

    def spread2_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket2_position: int,
        spread_data: Dict[str, Any],
    ):
        if Product.PICNIC_BASKET2 not in order_depths.keys():
            return None

        basket2_order_depth = order_depths[Product.PICNIC_BASKET2]
        synthetic2_order_depth = self.get_synthetic2_basket_order_depth(order_depths)
        basket2_swmid = self.get_swmid(basket2_order_depth)
        synthetic2_swmid = self.get_swmid(synthetic2_order_depth)
        spread2 = basket2_swmid - synthetic2_swmid
        spread_data.setdefault("spread_history").append(spread2)

        if (
            len(spread_data["spread_history"])
            < self.params[Product.SPREAD2]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD2]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread2_std = np.std(spread_data["spread_history"])

        zscore = (
            spread2 - self.params[Product.SPREAD2]["default_spread_mean"]
        ) / spread2_std

        if zscore >= self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket2_position != -self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread2_orders(
                    -self.params[Product.SPREAD2]["target_position"],
                    basket2_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket2_position != self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread2_orders(
                    self.params[Product.SPREAD2]["target_position"],
                    basket2_position,
                    order_depths,
                )

        spread_data["prev_zscore"] = zscore
        return None









    def rainforest_resin_orders(self, order_depth: OrderDepth, fair_value: int, width: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0
        # mm_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 20])
        # mm_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 20])
        
        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])
        
        # Take Orders
        buy_order_volume, sell_order_volume = self.take_best_orders(Product.RAINFOREST_RESIN, fair_value, 0.5, orders, order_depth, position, buy_order_volume, sell_order_volume)
        # Clear Position Orders
        buy_order_volume, sell_order_volume = self.clear_position_order(Product.RAINFOREST_RESIN, fair_value, 1, orders, order_depth, position, buy_order_volume, sell_order_volume)
        # Market Make
        buy_order_volume, sell_order_volume = self.market_make(Product.RAINFOREST_RESIN, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume)

        return orders
    

    def kelp_orders(self, order_depth: OrderDepth, timespan:int, width: float, kelp_take_width: float, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:    
            
            # Calculate Fair
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
            
            mmmid_price = (mm_ask + mm_bid) / 2    
            self.kelp_prices.append(mmmid_price)

            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.kelp_vwap.append({"vol": volume, "vwap": vwap})
            # self.KELP_mmmid.append(mmmid_price)
            
            if len(self.kelp_vwap) > timespan:
                self.kelp_vwap.pop(0)
            
            if len(self.kelp_prices) > timespan:
                self.kelp_prices.pop(0)
        
            # fair_value = sum([x["vwap"]*x['vol'] for x in self.kelp_vwap]) / sum([x['vol'] for x in self.kelp_vwap])=
            # fair_value = sum(self.kelp_prices) / len(self.kelp_prices)
            fair_value = mmmid_price

            # only taking best bid/ask
            buy_order_volume, sell_order_volume = self.take_best_orders(Product.KELP, fair_value, kelp_take_width, orders, order_depth, position, buy_order_volume, sell_order_volume, True, 50)
            
            # Clear Position Orders
            buy_order_volume, sell_order_volume = self.clear_position_order(Product.KELP, fair_value, 2, orders, order_depth, position, buy_order_volume, sell_order_volume)
            
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2

            # Market Make
            buy_order_volume, sell_order_volume = self.market_make(Product.KELP, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume)

        return orders
    
    def squid_ink_orders(self, order_depth: OrderDepth, timespan:int, width: float, squid_ink_take_width: float, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:    
            
            # Calculate Fair
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
            
            mmmid_price = (mm_ask + mm_bid) / 2    
            self.squid_ink_prices.append(mmmid_price)

            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.squid_ink_vwap.append({"vol": volume, "vwap": vwap})
            # self.squid_ink_mmmid.append(mmmid_price)
            
            if len(self.squid_ink_vwap) > timespan:
                self.squid_ink_vwap.pop(0)
            
            if len(self.squid_ink_prices) > timespan:
                self.squid_ink_prices.pop(0)
        
            # fair_value = sum([x["vwap"]*x['vol'] for x in self.squid_ink_vwap]) / sum([x['vol'] for x in self.squid_ink_vwap])=
            # fair_value = sum(self.squid_ink_prices) / len(self.squid_ink_prices)
            fair_value = mmmid_price

            # only taking best bid/ask
            buy_order_volume, sell_order_volume = self.take_best_orders(Product.SQUID_INK, fair_value, squid_ink_take_width, orders, order_depth, position, buy_order_volume, sell_order_volume, True, 50)
            
            # Clear Position Orders
            buy_order_volume, sell_order_volume = self.clear_position_order(Product.SQUID_INK, fair_value, 2, orders, order_depth, position, buy_order_volume, sell_order_volume)
            
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2

            # Market Make
            buy_order_volume, sell_order_volume = self.market_make(Product.SQUID_INK, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume)

        return orders





































    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        rainforest_resin_position_limit = 50

        kelp_make_width = 2.75#3.5
        kelp_take_width = 1
        kelp_position_limit = 50
        kelp_timemspan = 10
        
        squid_ink_make_width = 2.65#3.5
        squid_ink_take_width = 1
        squid_ink_position_limit = 50
        squid_ink_timemspan = 10


        if Product.RAINFOREST_RESIN in state.order_depths:      # additional and satement to be in self.params is missing ehre
            rainforest_resin_position = state.position[Product.RAINFOREST_RESIN] if Product.RAINFOREST_RESIN in state.position else 0
            rainforest_resin_orders = self.rainforest_resin_orders(state.order_depths[Product.RAINFOREST_RESIN], 
                                                                   self.params[Product.RAINFOREST_RESIN]["fair_value"], 
                                                                   self.params[Product.RAINFOREST_RESIN]["take_width"], 
                                                                   rainforest_resin_position, 
                                                                   rainforest_resin_position_limit)
            result[Product.RAINFOREST_RESIN] = rainforest_resin_orders

        if Product.KELP in state.order_depths:
            kelp_position = state.position[Product.KELP] if Product.KELP in state.position else 0
            kelp_orders = self.kelp_orders(state.order_depths[Product.KELP], 
                                           kelp_timemspan, 
                                           kelp_make_width, 
                                           kelp_take_width, 
                                           kelp_position, 
                                           kelp_position_limit)
            result[Product.KELP] = kelp_orders

        if Product.SQUID_INK in state.order_depths:
            squid_ink_position = state.position[Product.SQUID_INK] if Product.SQUID_INK in state.position else 0
            squid_ink_orders = self.squid_ink_orders(state.order_depths[Product.SQUID_INK], squid_ink_timemspan, 
                                                     squid_ink_make_width, 
                                                     squid_ink_take_width, 
                                                     squid_ink_position, 
                                                     squid_ink_position_limit)
            result[Product.SQUID_INK] = squid_ink_orders



        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            kelp_fair_value = self.kelp_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            kelp_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["take_width"],
                    kelp_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            kelp_make_orders, _, _ = self.make_kelp_orders(
                state.order_depths[Product.KELP],
                kelp_fair_value,
                self.params[Product.KELP]["kelp_min_edge"],
                kelp_position,
                buy_order_volume,
                sell_order_volume,
            )
            result[Product.KELP] = (
                kelp_take_orders + kelp_clear_orders + kelp_make_orders
            )

        # if Product.ORCHIDS in self.params and Product.ORCHIDS in state.order_depths:
        #     orchids_position = (
        #         state.position[Product.ORCHIDS]
        #         if Product.ORCHIDS in state.position
        #         else 0
        #     )
        #     print(f"ORCHIDS POSITION: {orchids_position}")

        #     conversions = self.orchids_arb_clear(orchids_position)

        #     orchids_position = 0

        #     orchids_take_orders, buy_order_volume, sell_order_volume = (
        #         self.orchids_arb_take(
        #             state.order_depths[Product.ORCHIDS],
        #             state.observations.conversionObservations[Product.ORCHIDS],
        #             orchids_position,
        #         )
        #     )

        #     orchids_make_orders, _, _ = self.orchids_arb_make(
        #         state.observations.conversionObservations[Product.ORCHIDS],
        #         orchids_position,
        #         buy_order_volume,
        #         sell_order_volume,
        #     )

        #     result[Product.ORCHIDS] = orchids_take_orders + orchids_make_orders

        if Product.SPREAD1 not in traderObject:
            traderObject[Product.SPREAD1] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket1_position = (
            state.position[Product.PICNIC_BASKET1]
            if Product.PICNIC_BASKET1 in state.position
            else 0
        )
        spread1_orders = self.spread1_orders(
            state.order_depths,
            Product.PICNIC_BASKET1,
            basket1_position,
            traderObject[Product.SPREAD1],
        )
        if spread1_orders != None:
            result[Product.CROISSANTS] = spread1_orders[Product.CROISSANTS]
            result[Product.JAMS] = spread1_orders[Product.JAMS]
            result[Product.DJEMBES] = spread1_orders[Product.DJEMBES]
            result[Product.PICNIC_BASKET1] = spread1_orders[Product.PICNIC_BASKET1]
        
        if Product.SPREAD2 not in traderObject:
            traderObject[Product.SPREAD2] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket2_position = (
            state.position[Product.PICNIC_BASKET2]
            if Product.PICNIC_BASKET2 in state.position
            else 0
        )
        spread2_orders = self.spread2_orders(
            state.order_depths,
            Product.PICNIC_BASKET2,
            basket2_position,
            traderObject[Product.SPREAD2],
        )
        if spread2_orders != None:
            result[Product.CROISSANTS] = spread2_orders[Product.CROISSANTS]
            result[Product.JAMS] = spread2_orders[Product.JAMS]
            result[Product.PICNIC_BASKET2] = spread2_orders[Product.PICNIC_BASKET2]
        
        # if Product.JAMS in state.order_depths:
        #     jams_position = state.position.get(Product.JAMS, 0)
        #     jams_orders = self.jams_orders(state.order_depths[Product.JAMS], jams_position)
        #     if Product.JAMS in result:
        #         result[Product.JAMS] += jams_orders  # combine with spread orders
        #     else:
        #         result[Product.JAMS] = jams_orders
        
    #     if Product.CROISSANTS in state.order_depths:
    #         croissants_position = state.position.get(Product.CROISSANTS, 0)
    #         result[Product.CROISSANTS] = self.croissants_orders(
    #             state.order_depths[Product.CROISSANTS],
    #             croissants_position
    # )





        traderData = jsonpickle.encode(traderObject)

        conversions = 1
        logger.flush(state,result,conversions,traderData)
        return result, conversions, traderData
    




