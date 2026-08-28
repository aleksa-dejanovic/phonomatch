"""HTTP server infrastructure with bounded request resources."""

import socket
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import BoundedSemaphore, Lock
from typing import cast

MAX_INFLIGHT_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_CONCURRENT_REQUESTS = 4
REQUEST_TIMEOUT_SECONDS = 15.0


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server with bounded active work and upload storage."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[BaseHTTPRequestHandler],
        *,
        max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS,
        max_inflight_upload_bytes: int = MAX_INFLIGHT_UPLOAD_BYTES,
        request_timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        if max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be positive")
        if max_inflight_upload_bytes < 1:
            raise ValueError("max_inflight_upload_bytes must be positive")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        super().__init__(server_address, request_handler_class)
        self.max_concurrent_requests = max_concurrent_requests
        self.max_inflight_upload_bytes = max_inflight_upload_bytes
        self.request_timeout_seconds = request_timeout_seconds
        self._request_slots = BoundedSemaphore(max_concurrent_requests)
        self._upload_lock = Lock()
        self._inflight_upload_bytes = 0

    def process_request(
        self,
        request: socket.socket | tuple[bytes, socket.socket],
        client_address: object,
    ) -> None:
        if not self._request_slots.acquire(blocking=False):
            self._send_overloaded_response(request)
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._request_slots.release()
            raise

    def process_request_thread(
        self,
        request: socket.socket | tuple[bytes, socket.socket],
        client_address: object,
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()

    def reserve_upload(self, size: int) -> bool:
        with self._upload_lock:
            if self._inflight_upload_bytes + size > self.max_inflight_upload_bytes:
                return False
            self._inflight_upload_bytes += size
            return True

    def release_upload(self, size: int) -> None:
        with self._upload_lock:
            self._inflight_upload_bytes -= size

    @staticmethod
    def _send_overloaded_response(
        request: socket.socket | tuple[bytes, socket.socket],
    ) -> None:
        body = b'{"error":"server is busy; try again later"}'
        response = (
            b"HTTP/1.1 503 Service Unavailable\r\n"
            b"Content-Type: application/json; charset=utf-8\r\n"
            + f"Content-Length: {len(body)}\r\n".encode()
            + b"Connection: close\r\n\r\n"
            + body
        )
        with suppress(OSError):
            cast(socket.socket, request).sendall(response)
