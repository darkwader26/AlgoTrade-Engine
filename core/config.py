"""Configuration system for the trading bot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class TradingConfig:
    """Trading bot configuration with sensible defaults."""

    # Trading parameters
    symbols: list[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD"])
    timeframes: list[str] = field(default_factory=lambda: ["1m", "5m", "1h"])
    initial_capital: float = 100_000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%

    # Position sizing
    position_sizing: Literal["fixed", "percent", "kelly"] = "fixed"
    max_position_size: float = 10_000.0

    # Risk management
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.05  # 5%
    max_drawdown_pct: float = 0.20  # 20%

    # Execution
    use_paper_trading: bool = True

    @classmethod
    def defaults(cls) -> TradingConfig:
        """Return a TradingConfig with default values."""
        return cls()


def load_config(path: str = "config.yaml") -> TradingConfig:
    """Load configuration from a YAML file.

    Falls back to defaults if the file does not exist or YAML is not installed.
    """
    try:
        import yaml
    except ImportError:
        return TradingConfig.defaults()

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return TradingConfig.defaults()

    with open(resolved, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        return TradingConfig.defaults()

    # Only pass known fields
    known_fields = set(TradingConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in data.items() if k in known_fields}
    return TradingConfig(**filtered)


def save_config(config: TradingConfig, path: str = "config.yaml") -> None:
    """Save a TradingConfig to a YAML file.

    Falls back to JSON if YAML is not installed.
    """
    try:
        import yaml
    except ImportError:
        import json

        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved, "w") as f:
            json.dump(asdict(config), f, indent=2)
        return

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)
