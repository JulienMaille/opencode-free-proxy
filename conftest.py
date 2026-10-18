"""Keep the offline suite out of the live proxy's files.

The pool tests exercise ``report_*`` helpers, and every one of those logs
through ``proxy_pool._log``, whose default target is ``data/proxy-pool.log`` —
the log of the running proxy. Running the suite therefore appended fake-exit
lines (test addresses such as ``9.9.9.9:4145``) to the live log. Point that
logger at a temp file for the whole session instead.

State files are already kept off disk by each pool test setting
``_last_stats_save``; this covers the log, which no test can suppress itself.
"""
import tempfile
from pathlib import Path

import proxy_pool

proxy_pool.log_file = (
    Path(tempfile.gettempdir()) / "opencode-free-proxy-tests" / "proxy-pool.log"
)
