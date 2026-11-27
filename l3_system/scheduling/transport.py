"""Bounded JSON request/reply over libzmq loopback TCP for the local C fixture."""
import ctypes as ct
from ctypes.util import find_library
import json
import socket


MAX_FRAME = 16_384


class TransportError(RuntimeError):
    pass


class ZmqEndpoint:
    """One process, one socket. Close and reconnect after a request times out."""

    def __init__(self, *, server, timeout_ms, address=None, max_frame=MAX_FRAME):
        if type(timeout_ms) is not int or not 0 < timeout_ms <= 60_000:
            raise ValueError("timeout_ms must be between 1 and 60000")
        if not server and (address is None or not address.startswith("tcp://127.0.0.1:")):
            raise ValueError("Client address must use IPv4 loopback")
        library = find_library("zmq")
        if not library:
            raise TransportError("libzmq is missing")
        if type(max_frame) is not int or not 0 < max_frame <= 1_048_576:
            raise ValueError("max_frame must be between 1 and 1048576")
        self.max_frame = max_frame
        self.lib = ct.CDLL(library)
        pointer, number, string = ct.c_void_p, ct.c_int, ct.c_char_p
        signatures = {
            "ctx_new": (pointer,), "ctx_term": (number, pointer),
            "socket": (pointer, pointer, number), "close": (number, pointer),
            "bind": (number, pointer, string), "connect": (number, pointer, string),
            "setsockopt": (number, pointer, number, pointer, ct.c_size_t),
            "send": (number, pointer, string, ct.c_size_t, number),
            "recv": (number, pointer, pointer, ct.c_size_t, number),
            "errno": (number,), "strerror": (string, number),
        }
        for name, (result, *arguments) in signatures.items():
            function = getattr(self.lib, f"zmq_{name}")
            function.restype = result
            function.argtypes = arguments
        self.context = self.lib.zmq_ctx_new()
        if not self.context:
            raise TransportError("Could not create ZeroMQ context")
        self.socket = self.lib.zmq_socket(self.context, 4 if server else 3)
        try:
            if not self.socket:
                raise TransportError("Could not create ZeroMQ socket")
            # libzmq socket option constants: LINGER, RCVTIMEO, SNDTIMEO.
            for option, value in ((17, 0), (27, timeout_ms), (28, timeout_ms)):
                setting = ct.c_int(value)
                self._check(self.lib.zmq_setsockopt(
                    self.socket, option, ct.byref(setting), ct.sizeof(setting)))
            max_frame = ct.c_int64(self.max_frame)
            self._check(self.lib.zmq_setsockopt(
                self.socket, 22, ct.byref(max_frame), ct.sizeof(max_frame)))  # ZMQ_MAXMSGSIZE
            if server:
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    listener.listen()
                    port = listener.getsockname()[1]
                    # ZMQ_USE_FD adopts a prebound loopback listener. The wildcard
                    # describes the libzmq endpoint and does not bind a second socket.
                    fd = ct.c_int(listener.fileno())
                    self._check(self.lib.zmq_setsockopt(
                        self.socket, 89, ct.byref(fd), ct.sizeof(fd)))
                    self._check(self.lib.zmq_bind(self.socket, f"tcp://*:{port}".encode()))
                    listener.detach()
                address = f"tcp://127.0.0.1:{port}"
            else:
                self._check(self.lib.zmq_connect(self.socket, address.encode()))
            self.address = address
        except BaseException:
            self.close()
            raise

    def _check(self, result):
        if result == -1:
            errno = self.lib.zmq_errno()
            raise TransportError(self.lib.zmq_strerror(errno).decode())
        return result

    def send(self, message):
        data = json.dumps(message, allow_nan=False).encode()
        if len(data) > self.max_frame:
            raise TransportError("Frame exceeds maximum size")
        self._check(self.lib.zmq_send(self.socket, data, len(data), 0))

    def receive(self):
        buffer = ct.create_string_buffer(self.max_frame)
        size = self._check(self.lib.zmq_recv(self.socket, buffer, self.max_frame, 0))
        if size > self.max_frame:
            raise TransportError("Frame exceeds maximum size")
        try:
            message = json.loads(buffer.raw[:size])
            json.dumps(message, allow_nan=False)
        except (ValueError, RecursionError) as error:
            raise TransportError("Invalid JSON frame") from error
        if not isinstance(message, dict):
            raise TransportError("Expected JSON object")
        return message

    def close(self):
        if self.socket:
            self.lib.zmq_close(self.socket)
            self.socket = None
        if self.context:
            self.lib.zmq_ctx_term(self.context)
            self.context = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
