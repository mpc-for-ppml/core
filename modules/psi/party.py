# modules/psi/party.py

from .ecc import generate_private_key, encrypt_point, hash_to_point

class Party:
    def __init__(self, name: str, dataset: list[str]):
        self.name = name
        self.dataset = dataset
        self.priv_key = generate_private_key()
        self.pub_set = [encrypt_point(hash_to_point(x), self.priv_key) for x in dataset]

    def re_encrypt(self, received_set: list[int]) -> list[int]:
        return [encrypt_point(point, self.priv_key) for point in received_set]

    def get_name(self):
        return self.name

    def get_dataset(self):
        return self.dataset

    def get_encrypted_set(self):
        return self.pub_set

    def get_private_key(self):
        return self.priv_key
    
    def compute_final_encrypted_items(self, all_parties):
        """Encrypt own dataset using all private keys, including self."""
        encrypted = [hash_to_point(x) for x in self.dataset]
        
        for party in all_parties:
            encrypted = [encrypt_point(p, party.get_private_key()) for p in encrypted]
        
        point_map = {
            (p.x, p.y): val
            for p, val in zip(encrypted, self.dataset)
        }
        return point_map
