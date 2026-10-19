"""Event notifications: the log, a recent-events buffer and an optional webhook."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Iterable
from typing import Any

import httpx

log = logging.getLogger(__name__)


class Notifier:
    def __init__(
        self,
        webhook_url: str | None = None,
        *,
        events: Iterable[str] | None = None,
        timeout_s: float = 5.0,
        client: httpx.Client | None = None,
        history: int = 100,
    ) -> None:
        self.webhook_url = webhook_url
        self.events = set(events) if events is not None else None
        self._client = client or httpx.Client(timeout=timeout_s)
        self.recent: deque[dict[str, Any]] = deque(maxlen=history)
        self._lock = threading.Lock()

    def notify(self, event: str, title: str, message: str, **data: Any) -> dict[str, Any]:
        """Record an event and, when configured for it, POST it to the webhook in the background.

        The JSON body has ``event``, ``title``, ``message``, ``data`` and ``timestamp``, plus a
        ``text`` field so chat webhooks (Slack, Mattermost) show it as a message.
        """
        payload = {
            "event": event,
            "title": title,
            "message": message,
            "text": f"{title}: {message}",
            "data": data,
            "timestamp": time.time(),
        }
        log.info("%s: %s", title, message)
        with self._lock:
            self.recent.append(payload)
        if self.webhook_url and (self.events is None or event in self.events):
            threading.Thread(target=self._post, args=(payload,), daemon=True).start()
        return payload

    def _post(self, payload: dict[str, Any]) -> None:
        assert self.webhook_url is not None
        try:
            self._client.post(self.webhook_url, json=payload).raise_for_status()
        except Exception as exc:  # a failed notification must never affect the machine
            log.warning("Webhook notification %r failed: %s", payload["event"], exc)

    def close(self) -> None:
        self._client.close()
