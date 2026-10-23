
import os
import logging
import threading
from datetime import datetime, timezone
from typing import Optional, Set

import groq
from supabase import create_client, Client

logger = logging.getLogger(__name__)


class SupabaseKeyManager:
    def __init__(self):
        url = os.getenv("SUPABASE_URL", "").strip()
        secret = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

        self.supabase: Optional[Client] = None
        self._state = threading.local()
        self._lock = threading.RLock()

        if not url or not secret:
            logger.error("Supabase credentials are missing.")
            return

        try:
            self.supabase = create_client(url, secret)
            logger.info("Groq key manager initialized.")
        except Exception:
            logger.exception("Supabase initialization failed.")

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    def get_current_key_id(self) -> Optional[str]:
        return getattr(self._state, "key_id", None)

    def _set_current(self, key_id=None, api_key=None):
        self._state.key_id = key_id
        self._state.api_key = api_key

    def refresh_keys_from_db(
        self,
        exclude_key_ids: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """Choose the least recently used active key, excluding attempted keys."""
        if self.supabase is None:
            logger.error("Cannot rotate keys: Supabase is unavailable.")
            self._set_current()
            return None

        exclude_key_ids = {
            str(key_id) for key_id in (exclude_key_ids or set())
        }

        with self._lock:
            try:
                query = (
                    self.supabase.table("groq_api_keys")
                    .select("id, api_key")
                    .eq("status", "active")
                )

                # Avoid retrying a key already tried in this request.
                for key_id in exclude_key_ids:
                    query = query.neq("id", key_id)

                response = (
                    query.order("last_used_at", desc=False)
                    .limit(1)
                    .execute()
                )

                rows = response.data or []

                if not rows:
                    self._set_current()
                    logger.warning(
                        "No untried active Groq keys remain."
                    )
                    return None

                row = rows[0]
                key_id = str(row.get("id", ""))
                api_key = (row.get("api_key") or "").strip()

                if not key_id or not api_key:
                    self._set_current()
                    logger.error("Selected key record is incomplete.")
                    return None

                # Mark this key as recently used.
                self.supabase.table("groq_api_keys").update(
                    {"last_used_at": self._now()}
                ).eq("id", key_id).execute()

                self._set_current(key_id, api_key)
                logger.info("Selected a Groq key for this request.")
                return api_key

            except Exception:
                self._set_current()
                logger.exception("Failed to select a Groq key.")
                return None

    def get_active_key(self) -> Optional[str]:
        return self.refresh_keys_from_db()

    def get_groq_client(
        self,
        exclude_key_ids: Optional[Set[str]] = None,
    ) -> Optional[groq.Groq]:
        """Create a client using a key not yet attempted in this request."""
        api_key = self.refresh_keys_from_db(exclude_key_ids)

        if not api_key:
            return None

        try:
            return groq.Groq(
                api_key=api_key,
                timeout=45.0,
                max_retries=0,
            )
        except Exception:
            logger.exception("Failed to initialize Groq client.")
            self._set_current()
            return None

    def rotate_key(
        self,
        exclude_key_ids: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """Select another active key without disabling the current key."""
        return self.refresh_keys_from_db(exclude_key_ids)

    def mark_key_unavailable(
        self,
        reason: str = "invalid_key",
    ) -> bool:
        """Disable the current key after a confirmed authentication failure."""
        key_id = self.get_current_key_id()

        if not key_id or self.supabase is None:
            logger.error("Cannot update key status.")
            self._set_current()
            return False

        try:
            self.supabase.table("groq_api_keys").update(
                {
                    "status": "exhausted",
                    "exhausted_at": self._now(),
                }
            ).eq("id", key_id).execute()

            logger.warning(
                "A Groq key was marked unavailable. Reason: %s",
                reason,
            )
            return True

        except Exception:
            logger.exception("Failed to update key status.")
            return False

        finally:
            self._set_current()

    def handle_quota_exhausted(self) -> Optional[str]:
        """
        Backward-compatible method.
        Call only when the key's quota is confirmed exhausted.
        """
        self.mark_key_unavailable(reason="quota_exhausted")
        return self.refresh_keys_from_db()
