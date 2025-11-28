from pathlib import Path
from datetime import datetime
import json
import numpy as np

from PLT_Regimes import get_nifty_regime_state
from PLT_Nifty_OptionTriggers import get_nifty_options_breakout_state


def _json_default(obj):
    """Handle datetimes / numpy types when dumping JSON."""
    from datetime import datetime as _dt
    if isinstance(obj, _dt):
        return obj.isoformat()

    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)

    return str(obj)


def build_snapshot() -> dict:
    regime_state = get_nifty_regime_state()
    options_state = get_nifty_options_breakout_state()

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "regime_state": regime_state,
        "options_state": options_state,
    }


def main():
    snap = build_snapshot()
    out_dir = Path("docs") / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nifty_snapshot.json"
    out_path.write_text(
        json.dumps(snap, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"Wrote snapshot -> {out_path}")


if __name__ == "__main__":
    main()
