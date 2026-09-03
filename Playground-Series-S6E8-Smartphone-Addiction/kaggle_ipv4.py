"""Run Kaggle CLI while filtering DNS results to IPv4 on this host."""
import socket
import sys

_original_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(*args, **kwargs):
    results = _original_getaddrinfo(*args, **kwargs)
    ipv4 = [result for result in results if result[0] == socket.AF_INET]
    return ipv4 or results


socket.getaddrinfo = _ipv4_getaddrinfo

from kaggle.cli import main

sys.exit(main())
