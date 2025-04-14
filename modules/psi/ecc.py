# modules/psi/ecc.py

import secrets
from tinyec import registry
from hashlib import sha256

curve = registry.get_curve("secp256r1")

def generate_private_key():
    # Generate secure random scalar within the curve order
    return secrets.randbelow(curve.field.n - 1) + 1

def hash_to_point(value: str):
    # Hash string to integer, then multiply with base point
    digest = sha256(value.encode()).hexdigest()
    int_val = int(digest, 16)
    return int_val * curve.g

def encrypt_point(point, private_scalar):
    return private_scalar * point

def point_to_bytes(point):
    return point.x.to_bytes(32, 'big') + point.y.to_bytes(32, 'big')

def bytes_to_point(b):
    x = int.from_bytes(b[:32], 'big')
    y = int.from_bytes(b[32:], 'big')
    return curve.point(x, y)
