import json
from typing import Any
from typing import List
import jsonpickle
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
        
        # Initialise traderData 
        traderData = state.traderData
        traderData = jsonpickle.decode(state.traderData) if state.traderData != "" else {}
        traderData.setdefault("prev_bid_ask", {})


        ###############
        for product in state.order_depths:
            # Common to all products 
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            POSITION_LIMIT = self.find_position_limits(product)
            
            logger.print(f"Product: {product}")
            
            
            
            if product == "RAINFOREST_RESIN": 
                """
                Market Making 
                
                - Account for missing bid/ask prices
                - Add previous bid/ask prices to traderData
                - Find best bid/ask prices based on volume
                - Find maximum market making quantity 
                - Send orders to market make 
                """
                #Check that bids/asks are NOT empty and calculate the market's best bid and best ask
                market_buy_orders = list(order_depth.buy_orders.items())
                market_sell_orders = list(order_depth.sell_orders.items())
                
                best_bid = market_buy_orders[0][0]
                best_ask = market_sell_orders[0][0]
                
                
                
                if state.timestamp != 0: # Won't have previous data at time = 0
                    last_bid = traderData['prev_bid_ask'][product][-1][0]
                    last_ask = traderData['prev_bid_ask'][product][-1][1]
                
                    #1 is an arbitrary number. We could find the average best bid/ask volume for a better estimate. Number should be small however to represent low interest. (ignore time 0)
                    if len(market_buy_orders) == 0 and len(market_sell_orders) != 0: #Bid orders empty
                        #Bid orders are empty. Use last and current ask price to estimate bid price
                        market_buy_orders = [last_bid - 1, 1]
                
                    elif len(market_buy_orders) == 0 and len(market_sell_orders) != 0: #Ask orders empty
                        #Ask orders empty. Use last and current bit price to estimate ask price
                        market_sell_orders = [last_ask + 1, 1]
                        
                    elif len(market_buy_orders) == 0 and len(market_sell_orders) == 0: #Both bids and asks empty
                        #Both orders empty.
                        market_buy_orders = [last_bid - 1, 1]
                        market_sell_orders = [last_ask + 1, 1]
                
                
                
                
                # Add data previous bid/ask to trader data 
                if product not in traderData["prev_bid_ask"]:
                    traderData["prev_bid_ask"][product] = [[best_bid, best_ask]]
                traderData = self.append_last_x_bid_ask_prices(traderData, product, market_buy_orders[0], market_sell_orders[0])
                
                
                
                # Find my best bid and ask 
                best_bid_vol = market_buy_orders[0][1]
                best_ask_vol = market_sell_orders[0][1]
                low_volume = {0, 1}
                
                #If the best bid and ask volume is low, we can match the price with higher volume to make more profit.
                """"
                Need something to account for large gaps in bid price like 10,005 then suddently 10,001
                """
                # Apply volume + gap logic
                if best_bid_vol in low_volume and abs(best_ask_vol) in low_volume:  # Both low
                    logger.print("(!!!) Both volume low (!!!) ")
                    my_bid = self.adjust_bid_with_gap(market_buy_orders, best_bid)
                    my_ask = self.adjust_ask_with_gap(market_sell_orders, best_ask)

                elif best_bid_vol in low_volume and abs(best_ask_vol) not in low_volume:  # Only bid low
                    logger.print("(!!!) Only bid volume low (!!!) ")
                    my_bid = self.adjust_bid_with_gap(market_buy_orders, best_bid)
                    my_ask = best_ask - 1

                elif best_bid_vol not in low_volume and abs(best_ask_vol) in low_volume:  # Only ask low
                    logger.print("(!!!) Only ask volume low (!!!) ")
                    my_bid = best_bid + 1
                    my_ask = self.adjust_ask_with_gap(market_sell_orders, best_ask)

                else:  # Both high volume
                    logger.print("(!!!) Both volume high (!!!) ")
                    my_bid = best_bid + 1
                    my_ask = best_ask - 1
                                        
                
                
                # Determine market making quantity 
                logger.print(f"Current Position: {state.position.get(product,0)}")
                if state.position.get(product, 0) == 0: # No positions
                    qty_bid = abs(POSITION_LIMIT)
                    qty_ask = abs(POSITION_LIMIT)
                elif state.position.get(product) > 0: #We are long
                    qty_bid = abs(POSITION_LIMIT) - state.position.get(product)
                    qty_ask = abs(POSITION_LIMIT) + state.position.get(product)
                else: # We are short
                    qty_bid = abs(POSITION_LIMIT) - state.position.get(product)
                    qty_ask = abs(POSITION_LIMIT) + state.position.get(product)
                logger.print(f"qty_bid: {qty_bid}")
                logger.print(f"qty_ask: {qty_ask}")
                
                
                # Send order to market make (just testing for now)
                if qty_bid != 0 or qty_ask != 0 and state.timestamp != 0: 
                    orders.append(Order(product, my_bid, qty_bid))
                    orders.append(Order(product, my_ask, -qty_ask))
                    # logger.print(f"Order sent: {product} {my_bid} {qty_bid}")
                    # logger.print(f"Order sent: {product} {my_ask} {-qty_ask}")
                
                
                
                
                
                
                
                
            
            elif product == "KELP": 
                pass 
            
            
            
            
            
            else: 
                logger.print("!!! No products found !!!")
            
    
    
    
    
            ################
            result[product] = orders
        logger.flush(state, result, conversions, jsonpickle.encode(traderData))
        return result, conversions, jsonpickle.encode(traderData)
    
    
    
    
    def find_position_limits(self, product) -> int:
        """
        For each product, find the position limited set (hard coded).
        """
        #Set position limits 
        product_limits = {'KELP': 50, 'RAINFOREST_RESIN': 50} 
        return product_limits[product]
        
        
    
    def append_last_x_bid_ask_prices(self, traderData, product, best_buy_order, best_ask_order):
        """
        For a particular product, update traderData to contain the last x values of the best bid and best ask.
        Used to calculate a fair value if a mid price is not given. 

        Change "data_hist" to the number of data periods you want to retain.
        """
        data_hist = 2

        if product not in traderData["prev_bid_ask"]:
            traderData["prev_bid_ask"][product] = [[best_buy_order[0], best_ask_order[0]]]
        elif len(traderData["prev_bid_ask"][product]) < data_hist:
            traderData["prev_bid_ask"][product].append([best_buy_order[0], best_ask_order[0]])
        else: 
            traderData["prev_bid_ask"][product].pop(0)
            traderData["prev_bid_ask"][product].append([best_buy_order[0], best_ask_order[0]])
        
        return traderData
    

    def adjust_bid_with_gap(self, buy_orders, best_bid):
        if len(buy_orders) > 1:
            second_best = buy_orders[1][0]
            gap = best_bid - second_best
            if gap >= 2:
                logger.print(f"Gap in bids detected: {gap}")
                return second_best + 1  # Beat second-best by 1
        return best_bid

    def adjust_ask_with_gap(self, sell_orders, best_ask):
        if len(sell_orders) > 1:
            second_best = sell_orders[1][0]
            gap = second_best - best_ask
            if gap >= 2:
                logger.print(f"Gap in asks detected: {gap}")
                return second_best - 1  # Beat second-best by 1
        return best_ask
