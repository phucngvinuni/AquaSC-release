"""Network-latency receiver (run on the PC). Persistent TCP connection; per message: 4-byte big-endian length N,
N payload bytes, then 1-byte ack back. Only accepts --allow IP. Exits after --idle seconds without a connection.
  python3 net_latency_receiver.py --bind PC_IP --port 5201 --allow LAPTOP_IP
"""
import argparse, socket, struct, time

def recvn(c, n):
    buf = bytearray()
    while len(buf) < n:
        d = c.recv(min(1 << 20, n - len(buf)))
        if not d:
            return None
        buf += d
    return bytes(buf)

ap = argparse.ArgumentParser()
ap.add_argument("--bind", required=True); ap.add_argument("--port", type=int, default=5201)
ap.add_argument("--allow", required=True); ap.add_argument("--idle", type=float, default=900)
a = ap.parse_args()
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((a.bind, a.port)); s.listen(1); s.settimeout(a.idle)
print("listening", a.bind, a.port, "allow", a.allow, flush=True)
while True:
    try:
        c, addr = s.accept()
    except socket.timeout:
        print("idle timeout, exit"); break
    if addr[0] != a.allow:
        print("rejected", addr[0], flush=True); c.close(); continue
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("connected", addr[0], flush=True); n_msg = 0
    while True:
        h = recvn(c, 4)
        if h is None: break
        n = struct.unpack(">I", h)[0]
        if recvn(c, n) is None: break
        c.sendall(b"\x06"); n_msg += 1
    print("closed after", n_msg, "messages", flush=True); c.close()
