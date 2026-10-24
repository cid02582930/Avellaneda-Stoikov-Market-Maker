from dataclasses import dataclass
import numpy as np
import pandas as pd

# params and config
@dataclass(frozen=True)
class MarketConfig:
    s0: float = 100.0                # initial mid price
    sigma: float = 0.15              # price volatility annualized converted to tick steps
    tick_size: float = 0.01          # minimum price increment
    dt: float = 1.0                  # discrete tick interval (seconds)
    total_time: float = 1800.0       # 30-minute trading session (seconds)
    
    # Hawkes Order Flow Process parameters
    mu_base: float = 0.8             # baseline market order intensity (events/sec)
    alpha_self: float = 1.2          # self-excitation coefficient
    alpha_cross: float = 0.3         # cross-excitation coefficient
    beta_decay: float = 2.0          # exponential decay parameter
    
    # microstructure execution parameters
    book_depth_lambda: float = 1.5   # decay rate of fill probability with distance kappa
    arrival_scale_A: float = 1.4     # fill arrival multiplier A
    mean_market_order_size: int = 5  # mean size of incoming aggressive market orders
    market_queue_depth: int = 15     # average passive shares sitting ahead on L2 book


@dataclass(frozen=True)
class MMConfig:
    gamma: float = 0.1               # CARA absolute risk aversion coefficient
    inventory_limit: int = 20        # hard stop inventory constraint
    quote_size: int = 1              # limit order quoting unit


# multivaruate Hawkes generator
class BivariateHawkesProcess:
    """
    Simulates cross- and self-exciting market order arrivals (buys & sells)
    using Ogata's Modified Thinning Algorithm
    """
    def __init__(self, config: MarketConfig, seed: int = 42):
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        
        # branching ratio check
        spec_radius = (self.cfg.alpha_self + self.cfg.alpha_cross) / self.cfg.beta_decay
        if spec_radius >= 1.0:
            raise ValueError(f"Hawkes process non-stationary. Spectral radius = {spec_radius:.3f} >= 1.0")

    def generate_events(self) -> pd.DataFrame:
        """
        Simulates arrivals over [0, total_time].
        Returns DataFrame containing arrival times, order directions and order volume
        """
        t = 0.0
        lambda_b = self.cfg.mu_base
        lambda_a = self.cfg.mu_base
        
        event_times = []
        event_types = []  # +1 for Market Buy, -1 for Market Sell
        
        while t < self.cfg.total_time:
            lambda_bar = lambda_b + lambda_a
            u = self.rng.uniform(0.0, 1.0)
            w = -np.log(u) / lambda_bar
            t += w
            
            if t >= self.cfg.total_time:
                break
                
            # decay intensities to candidate time t
            decay = np.exp(-self.cfg.beta_decay * w)
            lambda_b = self.cfg.mu_base + (lambda_b - self.cfg.mu_base) * decay
            lambda_a = self.cfg.mu_base + (lambda_a - self.cfg.mu_base) * decay
            
            # recompute total intensity at t
            lambda_current = lambda_b + lambda_a
            d = self.rng.uniform(0.0, 1.0)
            
            if d <= lambda_b / lambda_bar:
                # Market Buy accepted
                event_times.append(t)
                event_types.append(1)
                # self-excitation on buy, cross-excitation on sell
                lambda_b += self.cfg.alpha_self
                lambda_a += self.cfg.alpha_cross
            elif d <= lambda_current / lambda_bar:
                # Market Sell accepted
                event_times.append(t)
                event_types.append(-1)
                # self-excitation on sell, cross-excitation on buy
                lambda_a += self.cfg.alpha_self
                lambda_b += self.cfg.alpha_cross
            else:
                # event rejected by thinning step
                pass

        sizes = self.rng.geometric(p=1.0 / self.cfg.mean_market_order_size, size=len(event_times))
        
        return pd.DataFrame({
            "timestamp": event_times,
            "direction": event_types,
            "volume": sizes
        }).sort_values("timestamp").reset_index(drop=True)


# high-frequence order book & queue tracker
class PassiveLimitOrder:
    """Tracks position of our limit order inside a discrete FIFO book queue"""
    def __init__(self, price: float, size: int, queue_ahead: int):
        self.price = price
        self.size = size
        self.queue_ahead = queue_ahead
        self.is_active = True

    def process_trade(self, volume: int) -> int:
        """
        Processes an aggressive trade through the level
        Returns:
            shares_filled (int)
        """
        if not self.is_active:
            return 0
            
        if volume <= self.queue_ahead:
            self.queue_ahead -= volume
            return 0
        else:
            residual = volume - self.queue_ahead
            self.queue_ahead = 0
            fill = min(self.size, residual)
            self.size -= fill
            if self.size == 0:
                self.is_active = False
            return fill


# trading agents: AVELLANEDA-STOIKOV VS. NAIVE BENCHMARK
class AvellanedaStoikovAgent:
    """
    Optimal control market maker solving HJB pricing with inventory skew
    """
    def __init__(self, mm_cfg: MMConfig, mkt_cfg: MarketConfig):
        self.cfg = mm_cfg
        self.mkt = mkt_cfg

    def compute_quotes(self, s: float, q: int, t: float) -> tuple[float, float]:
        """
        Returns (bid_price, ask_price) snapped to ticks
        """
        time_left = max(self.mkt.total_time - t, 0.0) / self.mkt.total_time
        vol = self.mkt.sigma
        
        # 1. indifference reservation price r(s, q, t)
        reservation_price = s - q * self.cfg.gamma * (vol ** 2) * time_left
        
        # 2. optimal spread half-width
        half_spread = (1.0 / self.cfg.gamma) * np.log(1.0 + (self.cfg.gamma / self.mkt.book_depth_lambda))
        
        # 3. dynamic adjustment if inventory reaches limits
        if q >= self.cfg.inventory_limit:
            # Saturated long: widen ask, pull bid substantially
            bid_quote = reservation_price - 2.5 * half_spread
            ask_quote = reservation_price + 0.5 * half_spread
        elif q <= -self.cfg.inventory_limit:
            # saturated short: pull ask, tighten bid
            bid_quote = reservation_price - 0.5 * half_spread
            ask_quote = reservation_price + 2.5 * half_spread
        else:
            bid_quote = reservation_price - half_spread
            ask_quote = reservation_price + half_spread
            
        # 4. snap to discrete exchange tick size
        bid_quote = np.floor(bid_quote / self.mkt.tick_size) * self.mkt.tick_size
        ask_quote = np.ceil(ask_quote / self.mkt.tick_size) * self.mkt.tick_size
        
        # enforce non-crossing and minimum spread
        if bid_quote >= ask_quote:
            bid_quote = ask_quote - self.mkt.tick_size

        return float(bid_quote), float(ask_quote)


class NaiveSymmetricAgent:
    """
    Benchmark agent that maintains a constant spread centered symmetrically
    on the mid-price without inventory skewing
    """
    def __init__(self, half_spread_ticks: int, mkt_cfg: MarketConfig):
        self.half_spread = half_spread_ticks * mkt_cfg.tick_size
        self.mkt = mkt_cfg

    def compute_quotes(self, s: float, q: int, t: float) -> tuple[float, float]:
        bid_quote = np.floor((s - self.half_spread) / self.mkt.tick_size) * self.mkt.tick_size
        ask_quote = np.ceil((s + self.half_spread) / self.mkt.tick_size) * self.mkt.tick_size
        if bid_quote >= ask_quote:
            bid_quote = ask_quote - self.mkt.tick_size
        return float(bid_quote), float(ask_quote)


# discrete-event simulation engine
class SimulationEngine:
    def __init__(self, mkt_cfg: MarketConfig, mm_cfg: MMConfig, seed: int = 100):
        self.mkt = mkt_cfg
        self.mm = mm_cfg
        self.rng = np.random.default_rng(seed)

    def run(self, agent_type: str = "AS") -> dict:
        """
        Executes discrete-event matching loop
        agent_type: "AS" (Avellaneda-Stoikov) or "NAIVE" (Symmetric Fixed Spread)
        """
        # generate market order flow
        hawkes = BivariateHawkesProcess(self.mkt, seed=self.rng.integers(1, 100000))
        flow = hawkes.generate_events()
        
        # discretize mid-price trajectory via arithmetic Brownian motion
        n_steps = int(self.mkt.total_time / self.mkt.dt) + 1
        time_grid = np.linspace(0.0, self.mkt.total_time, n_steps)
        
        # volatility scaled per tick interval
        dt = self.mkt.dt
        dW = self.rng.normal(0.0, np.sqrt(dt), size=n_steps)
        vol_step = self.mkt.sigma * np.sqrt(dt / self.mkt.total_time)
        mid_prices = self.mkt.s0 + np.cumsum(vol_step * dW)
        mid_prices[0] = self.mkt.s0

        # Agent instantiation
        if agent_type == "AS":
            agent = AvellanedaStoikovAgent(self.mm, self.mkt)
        else:
            agent = NaiveSymmetricAgent(half_spread_ticks=2, mkt_cfg=self.mkt)

        # state tracking arrays
        current_inventory = 0
        current_cash = 0.0
        
        log_timestamps = []
        log_mid = []
        log_cash = []
        log_inventory = []
        log_pnl = []
        log_bid_quotes = []
        log_ask_quotes = []
        
        # active passive orders
        my_bid = None
        my_ask = None
        
        # netric tracking: Markout lists [(direction, fill_time, fill_price)]
        trade_executions = []
        
        flow_idx = 0
        n_flow_events = len(flow)
        
        for step, t in enumerate(time_grid):
            s_t = mid_prices[step]
            
            # 1. update MM quotes for current state
            bid_price, ask_price = agent.compute_quotes(s_t, current_inventory, t)
            
            # place or update passive limit orders with realistic queue depth ahead
            queue_depth_b = int(self.rng.poisson(self.mkt.market_queue_depth))
            queue_depth_a = int(self.rng.poisson(self.mkt.market_queue_depth))
            
            my_bid = PassiveLimitOrder(bid_price, self.mm.quote_size, queue_depth_b)
            my_ask = PassiveLimitOrder(ask_price, self.mm.quote_size, queue_depth_a)
            
            # 2. process all market orders occurring within interval [t, t + dt)
            while flow_idx < n_flow_events and flow.loc[flow_idx, "timestamp"] < t + dt:
                order_time = flow.loc[flow_idx, "timestamp"]
                direction = flow.loc[flow_idx, "direction"]
                vol = int(flow.loc[flow_idx, "volume"])
                flow_idx += 1
                
                # check execution against our book
                if direction == 1:
                    # aggressive market buy attempts to lift our ask
                    # arrival fills if order reaches/surpasses our ask level
                    level_penetration = self.rng.exponential(scale=1.0 / self.mkt.book_depth_lambda)
                    best_ask = s_t + self.mkt.tick_size
                    
                    if (best_ask + level_penetration) >= my_ask.price:
                        filled_shares = my_ask.process_trade(vol)
                        if filled_shares > 0:
                            current_inventory -= filled_shares
                            current_cash += filled_shares * my_ask.price
                            trade_executions.append({
                                "time": order_time,
                                "direction": -1,  # market maker sold
                                "fill_price": my_ask.price,
                                "step": step
                            })
                            
                elif direction == -1:
                    # aggressive market sell attempts to hit our bid
                    level_penetration = self.rng.exponential(scale=1.0 / self.mkt.book_depth_lambda)
                    best_bid = s_t - self.mkt.tick_size
                    
                    if (best_bid - level_penetration) <= my_bid.price:
                        filled_shares = my_bid.process_trade(vol)
                        if filled_shares > 0:
                            current_inventory += filled_shares
                            current_cash -= filled_shares * my_bid.price
                            trade_executions.append({
                                "time": order_time,
                                "direction": 1,  # market maker bought
                                "fill_price": my_bid.price,
                                "step": step
                            })
                            
            # 3. mark to Market PnL calculation
            mtm_pnl = current_cash + current_inventory * s_t
            
            log_timestamps.append(t)
            log_mid.append(s_t)
            log_cash.append(current_cash)
            log_inventory.append(current_inventory)
            log_pnl.append(mtm_pnl)
            log_bid_quotes.append(bid_price)
            log_ask_quotes.append(ask_price)

        history = pd.DataFrame({
            "timestamp": log_timestamps,
            "mid_price": log_mid,
            "cash": log_cash,
            "inventory": log_inventory,
            "pnl": log_pnl,
            "bid": log_bid_quotes,
            "ask": log_ask_quotes
        })
        
        trades_df = pd.DataFrame(trade_executions) if trade_executions else pd.DataFrame(
            columns=["time", "direction", "fill_price", "step"]
        )
        
        return {
            "history": history,
            "trades": trades_df,
            "mid_series": mid_prices,
            "time_grid": time_grid
        }


# Markout & PnL decomposition engine
class MarketMakingAnalytics:
    @staticmethod
    def calculate_markout(results: dict, horizons: list[int]) -> pd.DataFrame:
        """
        Computes conditional trade markout:
        E[Sign * (S_{t+tau} - S_fill)] for tau in discrete horizons
        """
        trades = results["trades"]
        mid = results["mid_series"]
        n_steps = len(mid)
        
        if trades.empty:
            return pd.DataFrame()
            
        markout_records = []
        for _, trade in trades.iterrows():
            step = int(trade["step"])
            direction = trade["direction"]  # +1 bought, -1 sold
            fill_p = trade["fill_price"]
            
            row = {"time": trade["time"], "direction": direction}
            for h in horizons:
                target_step = step + h
                if target_step < n_steps:
                    future_s = mid[target_step]
                    # Markout: positive indicates favorable edge
                    pnl_h = direction * (future_s - fill_p)
                else:
                    pnl_h = direction * (mid[-1] - fill_p)
                row[f"markout_{h}s"] = pnl_h
            markout_records.append(row)
            
        return pd.DataFrame(markout_records)

    @staticmethod
    def decompose_pnl(results: dict) -> dict:
        """
        Decomposes terminal PnL into:
        1. Pure Spread Capture (Cash flow if price had remained static)
        2. Inventory Mark-to-Market PnL
        3. Adverse Selection Drag
        """
        history = results["history"]
        trades = results["trades"]
        
        terminal_pnl = history["pnl"].iloc[-1]
        
        if trades.empty:
            return {
                "Terminal_PnL": 0.0,
                "Spread_Capture": 0.0,
                "Inventory_PnL": 0.0,
                "Trade_Count": 0
            }
            
        # spread capture: sum over all fills of |Fill Price - Mid at fill|
        spread_captured = 0.0
        for _, tr in trades.iterrows():
            step = int(tr["step"])
            mid_at_fill = history["mid_price"].iloc[step]
            fill_price = tr["fill_price"]
            spread_captured += abs(fill_price - mid_at_fill)
            
        inv_pnl = terminal_pnl - spread_captured
        
        # risk metrics
        returns = history["pnl"].diff().dropna()
        sharpe = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(len(returns))
        max_dd = (history["pnl"].cummax() - history["pnl"]).max()
        
        return {
            "Terminal_PnL ($)": round(terminal_pnl, 2),
            "Pure_Spread_Capture ($)": round(spread_captured, 2),
            "Inventory_Drift_PnL ($)": round(inv_pnl, 2),
            "Total_Trades": len(trades),
            "Mean_Inventory": round(history["inventory"].mean(), 2),
            "Std_Inventory": round(history["inventory"].std(), 2),
            "Max_Inventory": int(history["inventory"].abs().max()),
            "Session_Sharpe_Ratio": round(sharpe, 3),
            "Max_Drawdown ($)": round(max_dd, 2)
        }

# execution benchmark & comparison
if __name__ == "__main__":
    market_config = MarketConfig()
    mm_config = MMConfig()
    
    print("=" * 80)
    print("RUNNING HIGH-FREQUENCY MARKET MAKING SIMULATION (AVELLANEDA-STOIKOV VS NAIVE)")
    print("=" * 80)
    
    sim = SimulationEngine(market_config, mm_config, seed=2026)
    
    # 1. run Avellaneda-Stoikov Agent
    res_as = sim.run(agent_type="AS")
    stats_as = MarketMakingAnalytics.decompose_pnl(res_as)
    
    # 2. run Naive Fixed-Spread Agent
    res_naive = sim.run(agent_type="NAIVE")
    stats_naive = MarketMakingAnalytics.decompose_pnl(res_naive)
    
    # print Comparative Performance Table
    comparison_df = pd.DataFrame([stats_as, stats_naive], index=["Avellaneda-Stoikov", "Naive (Constant Spread)"])
    print("\n--- PERFORMANCE SUMMARY ---")
    print(comparison_df.T.to_string())
    
