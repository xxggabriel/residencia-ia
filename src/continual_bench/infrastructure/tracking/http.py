"""Keep-alive HTTP transport with the ``urllib.request.urlopen`` calling convention.

``urlopen`` opens a new TCP+TLS connection for every request; against the
SageMaker MLflow endpoint that handshake dominated each tracking call. This
opener reuses pooled connections (urllib3, already required by botocore) and
maps failures onto the ``urllib.error`` exceptions the tracker handles.
"""

import io
import threading
import urllib.error


class _Response:
    def __init__(self, status, data, headers):
        self.status = status
        self._data = data
        self.headers = headers

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class PooledHttpOpener:
    def __init__(self, *, maxsize: int = 16):
        import urllib3

        self._urllib3 = urllib3
        self._pool = urllib3.PoolManager(maxsize=maxsize, block=False, retries=False)

    def __call__(self, request, timeout=None):
        try:
            response = self._pool.request(
                request.get_method(),
                request.full_url,
                body=request.data,
                headers=dict(request.header_items()),
                timeout=timeout,
                redirect=False,
                preload_content=True,
            )
        except (self._urllib3.exceptions.HTTPError, TimeoutError, OSError) as error:
            raise urllib.error.URLError(error) from error
        if response.status >= 400:
            raise urllib.error.HTTPError(
                request.full_url,
                response.status,
                response.reason,
                response.headers,
                io.BytesIO(response.data),
            )
        return _Response(response.status, response.data, response.headers)


_shared: PooledHttpOpener | None = None
_shared_lock = threading.Lock()


def shared_opener() -> PooledHttpOpener:
    """One connection pool per process, shared by every tracker."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = PooledHttpOpener()
        return _shared
