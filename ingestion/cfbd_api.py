from __future__ import annotations

import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import CFBD_API_KEY, CFBD_BASE_URL


class CFBDClient:
    """Client for the CollegeFootballData API."""

    def __init__(self) -> None:
        if not CFBD_API_KEY:
            raise RuntimeError(
                "CFBD_API_KEY is missing. "
                "Add it to the project's .env file."
            )

        self.base_url = CFBD_BASE_URL.rstrip("/")

        self.session = requests.Session()

        self.session.headers.update(
            {
                "Authorization": f"Bearer {CFBD_API_KEY}",
                "Accept": "application/json",
                "User-Agent": "CFB-Prediction-Centre/2026",
            }
        )

        # Retry temporary network/server failures automatically.
        retry_strategy = Retry(
            total=5,
            connect=5,
            read=5,
            status=5,
            backoff_factor=1.5,
            status_forcelist=[
                429,
                500,
                502,
                503,
                504,
            ],
            allowed_methods=[
                "GET",
            ],
            respect_retry_after_header=True,
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
        )

        self.session.mount(
            "https://",
            adapter,
        )

        self.session.mount(
            "http://",
            adapter,
        )

    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Perform a GET request against the CFBD API."""

        url = (
            f"{self.base_url}/"
            f"{endpoint.lstrip('/')}"
        )

        last_error: Exception | None = None

        for attempt in range(1, 4):

            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=60,
                )

                if not response.ok:
                    print()
                    print("❌ CFBD API ERROR")
                    print(
                        f"Status: {response.status_code}"
                    )
                    print(
                        f"URL: {response.url}"
                    )
                    print(
                        f"Response: {response.text}"
                    )

                response.raise_for_status()

                return response.json()

            except (
                requests.ConnectionError,
                requests.Timeout,
            ) as exc:

                last_error = exc

                if attempt >= 3:
                    raise

                wait_seconds = 2 ** attempt

                print(
                    f"⚠ API connection problem "
                    f"(attempt {attempt}/3). "
                    f"Retrying in {wait_seconds}s..."
                )

                time.sleep(
                    wait_seconds
                )

        if last_error is not None:
            raise last_error

        raise RuntimeError(
            "CFBD API request failed unexpectedly."
        )