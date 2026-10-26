"""Polite HTTP for the data scripts: user agent, pause between calls, retries with backoff."""

from __future__ import annotations

import http.client
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

USER_AGENT = "BrushStrokeResearch/0.1 (https://github.com/CrispyChicken2/BrushAI; research)"


class Http:
    """Polite HTTP with retries (the Met API answers 403 when it is called too fast)."""

    def __init__(self, insecure: bool = False, pause: float = 0.15) -> None:
        self.ctx = ssl._create_unverified_context() if insecure else None  # noqa: S323
        self.pause = pause

    def get(self, url: str, retries: int = 6) -> bytes:
        url = urllib.parse.quote(
            url, safe=":/?&=%+,;@~#"
        )  # spaces in file names (Met), no double encoding
        for attempt in range(retries):
            time.sleep(self.pause)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(  # noqa: S310
                    req, context=self.ctx, timeout=60
                ) as resp:
                    return resp.read()  # type: ignore[no-any-return]
            except urllib.error.HTTPError as err:
                if err.code in (403, 429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(2.0 * 2**attempt)
                    continue
                raise
            except (urllib.error.URLError, http.client.HTTPException, OSError):
                if attempt == retries - 1:
                    raise
                time.sleep(2.0 * 2**attempt)
        raise RuntimeError(url)

    def json(self, url: str) -> Any:
        return json.loads(self.get(url))
