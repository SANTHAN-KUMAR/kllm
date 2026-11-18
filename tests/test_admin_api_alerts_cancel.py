"""A launch the planner cancels must not vanish silently: it shows as an alert for 30 minutes."""
import time

from gateway.admin_api import compute_alerts


def test_cancelled_launch_raises_an_alert():
    now = time.time()
    state = {"scheduler": {"log": [{"ts": now - 60, "action": "cancel", "target": "kg6", "why": "too slow to load"},
                                   {"ts": now - 4000, "action": "cancel", "target": "kg1", "why": "too slow to load"},
                                   {"ts": now - 30, "action": "launch", "target": "kg7"}]}}
    cfg = {"accounts": [{"name": "kg6"}]}
    alerts = [a for a in compute_alerts(cfg, state, {}, True, now=now) if a["id"].startswith("cancel-")]
    assert len(alerts) == 1 and "kg6" in alerts[0]["text"] and alerts[0]["level"] == "warn"
