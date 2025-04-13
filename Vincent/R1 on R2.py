from datamodel import OrderDepth, UserId, TradingState, Order, Listing, Observation, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List
import string
import jsonpickle
import numpy as np
import math
from typing import Tuple
from typing import Any
import json

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 2750

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

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"

class Trader:
    def __init__(self):
        self.kelp_prices = []
        self.kelp_vwap = []
        self.kelp_mmmid = []
        
        self.squid_ink_prices = []
        self.squid_ink_vwap = []
        self.squid_ink_mmmid = []
        
        self.croissants_prices = []
        self.croissants_vwap = []
        self.croissants_mmmid = []
        
        self.djembes_prices = []
        self.djembes_vwap = []
        self.djembes_mmmid = []
        
        self.jams_prices = []
        self.jams_vwap = []
        self.jams_mmmid = []
        
        self.picnic_basket1_prices = []
        self.picnic_basket1_vwap = []
        self.picnic_basket1_mmmid = []
        
        self.picnic_basket2_prices = []
        self.picnic_basket2_vwap = []
        self.picnic_basket2_mmmid = []
       

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.CROISSANTS: 250,
            Product.JAMS: 350, 
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60, 
            Product.PICNIC_BASKET2: 100
            
        }

    # Returns buy_order_volume, sell_order_volume
    def take_best_orders(self, product: str, fair_value: int, take_width:float, orders: List[Order], order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int, prevent_adverse: bool = False, adverse_volume: int = 0) -> Tuple[int, int]:
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            if prevent_adverse:
                if best_ask_amount <= adverse_volume and best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position) # max amt to buy 
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity)) 
                        buy_order_volume += quantity
            else:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position) # max amt to buy 
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity)) 
                        buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if prevent_adverse:
                if (best_bid >= fair_value + take_width) and (best_bid_amount <= adverse_volume):
                    quantity = min(best_bid_amount, position_limit + position) # should be the max we can sell 
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity

            else:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position) # should be the max we can sell 
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity

        return buy_order_volume, sell_order_volume
    
    def market_make(self, product: str, orders: List[Order], bid: int, ask: int, position: int, buy_order_volume: int, sell_order_volume: int) -> Tuple[int, int]:
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, bid, buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, ask, -sell_quantity))  # Sell order
    
        
        return buy_order_volume, sell_order_volume
    
    def clear_position_order(self, product: str, fair_value: float, width: int, orders: List[Order], order_depth: OrderDepth, position: int, buy_order_volume: int, sell_order_volume: int) -> List[Order]:
        
        position_after_take = position + buy_order_volume - sell_order_volume
        fair = round(fair_value)
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)
        # fair_for_ask = fair_for_bid = fair

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                # clear_quantity = position_after_take
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                # clear_quantity = abs(position_after_take)
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
    
        return buy_order_volume, sell_order_volume

    def fair_value(self, order_depth: OrderDepth, method = "mid_price", min_vol = 0) -> float:
        if method == "mid_price":
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            return mid_price
        elif method == "mid_price_with_vol_filter":
            if len([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]) ==0 or len([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]) ==0:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                mid_price = (best_ask + best_bid) / 2
                return mid_price
            else:   
                best_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol])
                best_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol])
                mid_price = (best_ask + best_bid) / 2
            return mid_price
    """
    #################################################################################################################################################
    #################################################################################################################################################
    """

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
    
    def squid_ink_orders(self, order_depth: OrderDepth, timespan:int, width: float, squid_take_width: float, position: int, position_limit: int) -> List[Order]:
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
            buy_order_volume, sell_order_volume = self.take_best_orders(Product.SQUID_INK, fair_value, squid_take_width, orders, order_depth, position, buy_order_volume, sell_order_volume, True, 50)
            
            # Clear Position Orders
            buy_order_volume, sell_order_volume = self.clear_position_order(Product.SQUID_INK, fair_value, 2, orders, order_depth, position, buy_order_volume, sell_order_volume)
            
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2

            # Market Make
            buy_order_volume, sell_order_volume = self.market_make(Product.SQUID_INK, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume)

        return orders


    def croissants_orders(self, order_depth: OrderDepth, timespan:int, width: float, croissants_take_width: float, position: int, position_limit: int) -> List[Order]:
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
            self.croissants_prices.append(mmmid_price)

            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.croissants_vwap.append({"vol": volume, "vwap": vwap})
            # self.squid_ink_mmmid.append(mmmid_price)
            
            if len(self.squid_ink_vwap) > timespan:
                self.croissants_vwap.pop(0)
            
            if len(self.squid_ink_prices) > timespan:
                self.croissants_prices.pop(0)
        
            # fair_value = sum([x["vwap"]*x['vol'] for x in self.squid_ink_vwap]) / sum([x['vol'] for x in self.squid_ink_vwap])=
            # fair_value = sum(self.squid_ink_prices) / len(self.squid_ink_prices)
            fair_value = mmmid_price

            # only taking best bid/ask
            buy_order_volume, sell_order_volume = self.take_best_orders(Product.CROISSANTS, fair_value, croissants_take_width, orders, order_depth, position, buy_order_volume, sell_order_volume, True, 50)
            
            # Clear Position Orders
            buy_order_volume, sell_order_volume = self.clear_position_order(Product.CROISSANTS, fair_value, 2, orders, order_depth, position, buy_order_volume, sell_order_volume)
            
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2

            # Market Make
            buy_order_volume, sell_order_volume = self.market_make(Product.CROISSANTS, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume)

        return orders
    
    
    def jams_orders(self, order_depth: OrderDepth, timespan:int, width: float, jams_take_width: float, position: int, position_limit: int) -> List[Order]:
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
            self.jams_prices.append(mmmid_price)

            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.jams_vwap.append({"vol": volume, "vwap": vwap})
            # self.squid_ink_mmmid.append(mmmid_price)
            
            if len(self.squid_ink_vwap) > timespan:
                self.jams_vwap.pop(0)
            
            if len(self.squid_ink_prices) > timespan:
                self.jams_prices.pop(0)
        
            # fair_value = sum([x["vwap"]*x['vol'] for x in self.squid_ink_vwap]) / sum([x['vol'] for x in self.squid_ink_vwap])=
            # fair_value = sum(self.squid_ink_prices) / len(self.squid_ink_prices)
            fair_value = mmmid_price

            # only taking best bid/ask
            buy_order_volume, sell_order_volume = self.take_best_orders(Product.JAMS, fair_value, jams_take_width, orders, order_depth, position, buy_order_volume, sell_order_volume, True, 50)
            
            # Clear Position Orders
            buy_order_volume, sell_order_volume = self.clear_position_order(Product.JAMS, fair_value, 2, orders, order_depth, position, buy_order_volume, sell_order_volume)
            
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2

            # Market Make
            buy_order_volume, sell_order_volume = self.market_make(Product.JAMS, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume)

        return orders
    
    
    def djembes_orders(self, order_depth: OrderDepth, timespan:int, width: float, djembes_take_width: float, position: int, position_limit: int) -> List[Order]:
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
            self.djembes_prices.append(mmmid_price)

            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.djembes_vwap.append({"vol": volume, "vwap": vwap})
            # self.squid_ink_mmmid.append(mmmid_price)
            
            if len(self.squid_ink_vwap) > timespan:
                self.djembes_vwap.pop(0)
            
            if len(self.squid_ink_prices) > timespan:
                self.djembes_prices.pop(0)
        
            # fair_value = sum([x["vwap"]*x['vol'] for x in self.squid_ink_vwap]) / sum([x['vol'] for x in self.squid_ink_vwap])=
            # fair_value = sum(self.squid_ink_prices) / len(self.squid_ink_prices)
            fair_value = mmmid_price

            # only taking best bid/ask
            buy_order_volume, sell_order_volume = self.take_best_orders(Product.DJEMBES, fair_value, djembes_take_width, orders, order_depth, position, buy_order_volume, sell_order_volume, True, 50)
            
            # Clear Position Orders
            buy_order_volume, sell_order_volume = self.clear_position_order(Product.DJEMBES, fair_value, 2, orders, order_depth, position, buy_order_volume, sell_order_volume)
            
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2

            # Market Make
            buy_order_volume, sell_order_volume = self.market_make(Product.DJEMBES, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume)

        return orders
    
    def picnic_basket1_orders(self, order_depth: OrderDepth, timespan:int, width: float, picnic_basket1_take_width: float, position: int, position_limit: int) -> List[Order]:
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
            self.picnic_basket1_prices.append(mmmid_price)

            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.picnic_basket1_vwap.append({"vol": volume, "vwap": vwap})
            # self.squid_ink_mmmid.append(mmmid_price)
            
            if len(self.squid_ink_vwap) > timespan:
                self.picnic_basket1_vwap.pop(0)
            
            if len(self.squid_ink_prices) > timespan:
                self.picnic_basket1_prices.pop(0)
        
            # fair_value = sum([x["vwap"]*x['vol'] for x in self.squid_ink_vwap]) / sum([x['vol'] for x in self.squid_ink_vwap])=
            # fair_value = sum(self.squid_ink_prices) / len(self.squid_ink_prices)
            fair_value = mmmid_price

            # only taking best bid/ask
            buy_order_volume, sell_order_volume = self.take_best_orders(Product.PICNIC_BASKET1, fair_value, picnic_basket1_take_width, orders, order_depth, position, buy_order_volume, sell_order_volume, True, 50)
            
            # Clear Position Orders
            buy_order_volume, sell_order_volume = self.clear_position_order(Product.PICNIC_BASKET1, fair_value, 2, orders, order_depth, position, buy_order_volume, sell_order_volume)
            
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2

            # Market Make
            buy_order_volume, sell_order_volume = self.market_make(Product.PICNIC_BASKET1, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume)

        return orders
    
    
    def picnic_basket2_orders(self, order_depth: OrderDepth, timespan:int, width: float, picnic_basket2_take_width: float, position: int, position_limit: int) -> List[Order]:
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
            self.picnic_basket2_prices.append(mmmid_price)

            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.picnic_basket2_vwap.append({"vol": volume, "vwap": vwap})
            # self.squid_ink_mmmid.append(mmmid_price)
            
            if len(self.squid_ink_vwap) > timespan:
                self.picnic_basket2_vwap.pop(0)
            
            if len(self.squid_ink_prices) > timespan:
                self.picnic_basket2_prices.pop(0)
        
            # fair_value = sum([x["vwap"]*x['vol'] for x in self.squid_ink_vwap]) / sum([x['vol'] for x in self.squid_ink_vwap])=
            # fair_value = sum(self.squid_ink_prices) / len(self.squid_ink_prices)
            fair_value = mmmid_price

            # only taking best bid/ask
            buy_order_volume, sell_order_volume = self.take_best_orders(Product.PICNIC_BASKET2, fair_value, picnic_basket2_take_width, orders, order_depth, position, buy_order_volume, sell_order_volume, True, 50)
            
            # Clear Position Orders
            buy_order_volume, sell_order_volume = self.clear_position_order(Product.PICNIC_BASKET2, fair_value, 2, orders, order_depth, position, buy_order_volume, sell_order_volume)
            
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2

            # Market Make
            buy_order_volume, sell_order_volume = self.market_make(Product.PICNIC_BASKET2, orders, bbbf + 1, baaf - 1, position, buy_order_volume, sell_order_volume)

        return orders

    """
    #################################################################################################################################################
    #################################################################################################################################################
    """


    def run(self, state: TradingState):
        result = {}

        rainforest_resin_fair_value = 10000  # Participant should calculate this value
        rainforest_resin_width = 13.5#2
        rainforest_resin_position_limit = 50

        kelp_make_width = 2.75#3.5
        kelp_take_width = 1
        kelp_position_limit = 50
        kelp_timemspan = 10
        
        squid_ink_make_width = 2.65#3.5
        squid_ink_take_width = 1
        squid_ink_position_limit = 50
        squid_ink_timemspan = 10
        
        jams_make_width = 1.6
        jams_take_width = 0.65
        jams_position_limit = 350
        jams_timemspan = 10

        croissants_make_width = 1.2
        croissants_take_width = 0.5
        croissants_position_limit = 250
        croissants_timemspan = 10
        
        djembes_make_width = 1.25
        djembes_take_width = 0.5
        djembes_position_limit = 60
        djembes_timemspan = 10

        picnic_basket1_make_width = 30
        picnic_basket1_take_width = 12
        picnic_basket1_position_limit = 60
        picnic_basket1_timemspan = 10
 
        picnic_basket2_make_width = 30
        picnic_basket2_take_width = 12
        picnic_basket2_position_limit = 100
        picnic_basket2_timemspan = 10
        
        # traderData = jsonpickle.decode(state.traderData)
        # print(state.traderData)
        # self.kelp_prices = traderData["kelp_prices"]
        # self.kelp_vwap = traderData["kelp_vwap"]
        logger.print(state.traderData)

        if Product.RAINFOREST_RESIN in state.order_depths:
            rainforest_resin_position = state.position[Product.RAINFOREST_RESIN] if Product.RAINFOREST_RESIN in state.position else 0
            rainforest_resin_orders = self.rainforest_resin_orders(state.order_depths[Product.RAINFOREST_RESIN], rainforest_resin_fair_value, rainforest_resin_width, rainforest_resin_position, rainforest_resin_position_limit)
            result[Product.RAINFOREST_RESIN] = rainforest_resin_orders

        if Product.KELP in state.order_depths:
            kelp_position = state.position[Product.KELP] if Product.KELP in state.position else 0
            kelp_orders = self.kelp_orders(state.order_depths[Product.KELP], kelp_timemspan, kelp_make_width, kelp_take_width, kelp_position, kelp_position_limit)
            result[Product.KELP] = kelp_orders

        if Product.SQUID_INK in state.order_depths:
            squid_ink_position = state.position[Product.SQUID_INK] if Product.SQUID_INK in state.position else 0
            squid_ink_orders = self.squid_ink_orders(state.order_depths[Product.SQUID_INK], squid_ink_timemspan, squid_ink_make_width, squid_ink_take_width, squid_ink_position, squid_ink_position_limit)
            result[Product.SQUID_INK] = squid_ink_orders
        
        # Picnic 1
        if Product.PICNIC_BASKET1 in state.order_depths:
            picnic_basket1_position = state.position[Product.PICNIC_BASKET1] if Product.PICNIC_BASKET1 in state.position else 0
            picnic_basket1_orders = self.picnic_basket1_orders(state.order_depths[Product.PICNIC_BASKET1], picnic_basket1_timemspan, picnic_basket1_make_width, picnic_basket1_take_width, picnic_basket1_position, picnic_basket1_position_limit)
            result[Product.PICNIC_BASKET1] = picnic_basket1_orders
            
        # Picnic 2
        if Product.PICNIC_BASKET2 in state.order_depths:
            picnic_basket2_position = state.position[Product.PICNIC_BASKET2] if Product.PICNIC_BASKET2 in state.position else 0
            picnic_basket2_orders = self.picnic_basket2_orders(state.order_depths[Product.PICNIC_BASKET2], picnic_basket2_timemspan, picnic_basket2_make_width, picnic_basket2_take_width, picnic_basket2_position, picnic_basket2_position_limit)
            result[Product.PICNIC_BASKET2] = picnic_basket2_orders
            
        # Jam
        if Product.JAMS in state.order_depths:
            jams_position = state.position[Product.JAMS] if Product.JAMS in state.position else 0
            jams_orders = self.jams_orders(state.order_depths[Product.JAMS], jams_timemspan, jams_make_width, jams_take_width, jams_position, jams_position_limit)
            result[Product.JAMS] = jams_orders
            
        # Croissants
        if Product.CROISSANTS in state.order_depths:
            croissants_position = state.position[Product.CROISSANTS] if Product.CROISSANTS in state.position else 0
            croissants_orders = self.croissants_orders(state.order_depths[Product.CROISSANTS], croissants_timemspan, croissants_make_width, croissants_take_width, croissants_position, croissants_position_limit)
            result[Product.CROISSANTS] = croissants_orders
            
        # Djemebes
        if Product.DJEMBES in state.order_depths:
            djembes_position = state.position[Product.DJEMBES] if Product.DJEMBES in state.position else 0
            djembes_orders = self.djembes_orders(state.order_depths[Product.DJEMBES], djembes_timemspan, djembes_make_width, djembes_take_width, djembes_position, djembes_position_limit)
            result[Product.DJEMBES] = djembes_orders

        
        traderData = jsonpickle.encode( { 
                                         "kelp_prices": self.kelp_prices, "kelp_vwap": self.kelp_vwap, 
                                         "squid_ink_prices": self.squid_ink_prices, "squid_ink_vwap": self.squid_ink_vwap,
                                         "jams_prices": self.jams_prices, "jams_vwap": self.jams_vwap,
                                         "croissants_prices": self.croissants_prices, "croissants_vwap": self.croissants_vwap,
                                         "djembes_prices": self.djembes_prices, "djembes_vwap": self.djembes_vwap,
                                         "picnic_basket1_prices": self.picnic_basket1_prices, "picnic_basket1_vwap": self.picnic_basket1_vwap,
                                         "picnic_basket2_prices": self.picnic_basket2_prices, "picnic_basket2_vwap": self.picnic_basket2_vwap
                                         } )
        


        conversions = 1
        logger.flush(state,result,conversions,traderData)
        return result, conversions, traderData

    
