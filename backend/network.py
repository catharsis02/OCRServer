import socket

_original_socket = socket.socket

class GuardedSocket(_original_socket):
    def connect(self, *args, **kwargs):
        raise RuntimeError("Network disabled")

socket.socket = GuardedSocket # type : ignore