"""
VerifAI shared core.

Environment flags are set *here*, in the package `__init__`, rather than in
`core/config.py` — `huggingface_hub` reads HF_HUB_OFFLINE once at its own import
time, so anything that sets it after that import has no effect. Putting it in
`__init__` means `import core.<anything>` is early enough, and this module
deliberately has no heavy imports of its own so nothing can beat it to the punch.

Why offline at all: sentence-transformers checks the Hub for updated files on
every model load. On this machine that lookup crashes the process with a hard
access violation — see `init_native_libs` below for why, and note that Tier 1
has no business needing the network to rebuild an index from a local CSV anyway.
Set HF_OFFLINE=false only for a first run against a cold model cache.
"""

import logging
import os
import socket
import threading

logger = logging.getLogger(__name__)

if os.environ.get("HF_OFFLINE", "true").strip().lower() not in {"false", "0", "no"}:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ── DNS pre-resolution ───────────────────────────────────────────────────────
#
# Importing `sentence_transformers` leaves `socket.getaddrinfo` unable to run
# without killing the process (details in `init_native_libs`). Every outbound
# API call in Tier 2 goes through it — httpx/httpcore reach it via
# `socket.create_connection` — so the whole serving path would be unreachable.
#
# The fix is to resolve the hosts we will need *before* that import happens and
# answer from a cache afterwards, so the broken code path is never re-entered.
# Anything not in the cache still falls through to the real resolver, which
# keeps this from silently breaking hosts we did not anticipate.

# Hosts Tier 2 talks to. Overridable so a fork pointing at different endpoints
# does not have to edit code; set to an empty string to disable pre-resolution.
# translate.google.com is gTTS's endpoint for the M5 audio output; without it
# pre-resolved, the first spoken report kills the process.
_DEFAULT_HOSTS = (
    "api.groq.com,api.tavily.com,api.smith.langchain.com,translate.google.com"
)

# Total budget for pre-resolution. Lookups run concurrently, so a machine with
# no network costs this once rather than once per host — which matters because
# `ingest` calls this too and has no business waiting on DNS at all.
_PREWARM_TIMEOUT = 3.0

_dns_cache: dict[tuple[str, int], list] = {}
_real_getaddrinfo = socket.getaddrinfo
_dns_patched = False


def _prewarm_dns() -> None:
    """Resolve the known hosts and install a cache in front of getaddrinfo."""
    global _dns_patched
    if _dns_patched:
        return

    raw = os.environ.get("VERIFAI_DNS_PREWARM", _DEFAULT_HOSTS)
    hosts = [h.strip() for h in raw.split(",") if h.strip()]
    if not hosts:
        return

    def resolve(host: str) -> None:
        try:
            # SOCK_STREAM matches what socket.create_connection asks for, so the
            # cached entries are usable verbatim by every HTTP client here.
            _dns_cache[(host, 443)] = _real_getaddrinfo(
                host, 443, 0, socket.SOCK_STREAM
            )
        except OSError as exc:  # offline, or the host does not exist
            logger.debug("DNS pre-resolution failed for %s: %s", host, exc)

    threads = [
        threading.Thread(target=resolve, args=(h,), daemon=True) for h in hosts
    ]
    for thread in threads:
        thread.start()
    deadline = _PREWARM_TIMEOUT / max(len(threads), 1)
    for thread in threads:
        thread.join(timeout=deadline)

    def cached_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if isinstance(host, bytes):
            host = host.decode("ascii", errors="ignore")
        key = (host, port)
        if key in _dns_cache and type in (0, socket.SOCK_STREAM):
            return _dns_cache[key]
        return _real_getaddrinfo(host, port, family, type, proto, flags)

    socket.getaddrinfo = cached_getaddrinfo
    _dns_patched = True
    logger.debug("DNS pre-resolved: %s", sorted(h for h, _ in _dns_cache))


def init_native_libs() -> None:
    """
    Prepare the process for `sentence_transformers`, then import it.

    TWO SEPARATE WINDOWS FAULTS ARE HANDLED HERE, both presenting identically:
    the process dies with access violation 0xC0000005, no Python traceback, no
    MemoryError, nothing catchable. Both were found with `python -X faulthandler`,
    which prints a C-level stack where a normal run prints nothing at all.

    1. IMPORT ORDER. `chromadb` and `sentence_transformers` ship conflicting
       native runtimes (onnxruntime vs torch) and whichever initialises first
       wins:

           import sentence_transformers; import chromadb   -> works
           import chromadb; import sentence_transformers   -> 0xC0000005

       Ruled out: `KMP_DUPLICATE_LIB_OK=TRUE` (no effect), importing torch first
       (no effect — only the full sentence_transformers import works).

    2. DNS. Once `sentence_transformers` is imported, `socket.getaddrinfo`
       access-violates on *any* host. Verified by bisection:

           baseline                        -> resolves fine
           import torch                    -> resolves fine
           import sentence_transformers    -> CRASH

       Warming the OS resolver cache first does not help; the crash is inside
       the call, not in the lookup. So the hosts are resolved beforehand and
       served from `_dns_cache` afterwards.

       This is also the real explanation for the HF Hub crash seen during M1,
       which was 'fixed' by forcing offline mode — that removed the lookup
       rather than the fault.

    Call this at the top of any module that imports chromadb or loads a model.
    Idempotent; costs nothing after the first call.
    """
    _prewarm_dns()
    import sentence_transformers  # noqa: F401  (imported for its side effects)
