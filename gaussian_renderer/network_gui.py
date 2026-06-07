#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import socket
from errno import EWOULDBLOCK

conn = None
listener = None
addr = None


def init(wish_host, wish_port):
    global listener
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((wish_host, wish_port))
    listener.listen()
    listener.settimeout(0)


def try_connect():
    global conn, addr, listener
    if listener is None:
        return
    try:
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")
        conn.settimeout(None)
    except socket.error as inst:
        if inst.errno != EWOULDBLOCK:
            print(inst)


def read():
    global conn
    message_length = conn.recv(4)
    message_length = int.from_bytes(message_length, "little")
    return conn.recv(message_length)


def send(message_bytes, verify):
    global conn
    if message_bytes is not None:
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4, "little"))
    conn.sendall(bytes(verify, "ascii"))


def receive():
    # The viewer protocol is intentionally not reimplemented here because the
    # training path only needs a non-blocking connection shim when no viewer is
    # attached. If a viewer connects unexpectedly, close it cleanly.
    global conn
    if conn is not None:
        conn.close()
        conn = None
    return None, True, False, False, False, 1.0
