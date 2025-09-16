import os
import pickle
from typing import Iterator, List

import lmdb
import numpy as np

from ..selfplay.selfplay import SelfPlayData


class DataBuffer:
    """LMDB-backed replay buffer for training data"""

    def __init__(
        self, db_path: str, max_size: int = 5_000_000, map_size: int = 64 * 1024**3
    ):
        self.db_path = db_path
        self.max_size = max_size
        self.env = lmdb.open(db_path, map_size=map_size)  # Default 64GB
        self.size = 0
        self.write_idx = 0

        # Load existing size
        with self.env.begin() as txn:
            size_bytes = txn.get(b"size")
            if size_bytes:
                self.size = int.from_bytes(size_bytes, "big")
                self.write_idx = self.size % self.max_size

    def add_data(self, data: List[SelfPlayData]):
        """Add training examples with 8-fold symmetry augmentation"""
        augmented_data = []
        for example in data:
            augmented_data.extend(self._augment_example(example))

        try:
            with self.env.begin(write=True) as txn:
                for example in augmented_data:
                    key = f"data_{self.write_idx}".encode()
                    value = pickle.dumps(example)
                    txn.put(key, value)

                    self.write_idx = (self.write_idx + 1) % self.max_size
                    if self.size < self.max_size:
                        self.size += 1

                # Update size
                txn.put(b"size", self.size.to_bytes(8, "big"))
        except lmdb.MapFullError as e:
            print(f"⚠️  LMDB map full! Attempting to resize...")
            try:
                self._resize_map()
                # Retry with larger map
                with self.env.begin(write=True) as txn:
                    for example in augmented_data:
                        key = f"data_{self.write_idx}".encode()
                        value = pickle.dumps(example)
                        txn.put(key, value)

                        self.write_idx = (self.write_idx + 1) % self.max_size
                        if self.size < self.max_size:
                            self.size += 1

                    txn.put(b"size", self.size.to_bytes(8, "big"))
            except Exception as resize_error:
                print(f"❌ Failed to resize LMDB and retry operation: {resize_error}")
                raise RuntimeError(f"Unable to store data due to LMDB resize failure: {resize_error}") from e

    def _resize_map(self):
        """Dynamically resize LMDB map when full"""
        current_info = self.env.info()
        current_size = current_info["map_size"]
        new_size = int(current_size * 1.5)  # Increase by 50%

        print(f"📈 Resizing LMDB map: {current_size/1024**3:.1f}GB → {new_size/1024**3:.1f}GB")

        # Store backup reference to current environment
        old_env = self.env
        new_env = None

        try:
            # Attempt to open new environment with larger map size
            new_env = lmdb.open(self.db_path, map_size=new_size)

            # Only close old environment after successful opening
            old_env.close()
            self.env = new_env

            # Reload size info
            with self.env.begin() as txn:
                size_bytes = txn.get(b"size")
                if size_bytes:
                    self.size = int.from_bytes(size_bytes, "big")
                    self.write_idx = self.size % self.max_size

            print(f"✅ LMDB resize successful")

        except Exception as e:
            print(f"❌ LMDB resize failed: {e}")

            # Cleanup new environment if it was created
            if new_env is not None:
                try:
                    new_env.close()
                except:
                    pass

            # Ensure we still have a working environment
            if self.env != old_env:
                self.env = old_env

            # Re-raise the exception to be handled by the caller
            raise RuntimeError(f"Failed to resize LMDB map: {e}") from e

    def _augment_example(self, example: SelfPlayData) -> List[SelfPlayData]:
        """Apply 8-fold symmetry augmentation"""
        augmented = []
        state = example.state
        size = state.shape[-1]  # Board size from state shape
        policy = example.policy.reshape(size, size)

        for i in range(4):  # 4 rotations
            # Rotate state (all 5 channels)
            rot_state = np.rot90(state, i, axes=(1, 2))
            rot_policy = np.rot90(policy, i)

            augmented.append(
                SelfPlayData(
                    state=rot_state, policy=rot_policy.flatten(), value=example.value
                )
            )

            # Flip horizontally
            flip_state = np.flip(rot_state, axis=2)
            flip_policy = np.flip(rot_policy, axis=1)

            augmented.append(
                SelfPlayData(
                    state=flip_state, policy=flip_policy.flatten(), value=example.value
                )
            )

        return augmented

    def sample_batch(self, batch_size: int) -> List[SelfPlayData]:
        """Sample random batch from buffer"""
        if self.size == 0:
            return []

        indices = np.random.randint(0, self.size, batch_size)
        batch = []

        with self.env.begin() as txn:
            for idx in indices:
                key = f"data_{idx}".encode()
                value = txn.get(key)
                if value:
                    batch.append(pickle.loads(value))

        return batch

    def __len__(self) -> int:
        return self.size

    def close(self):
        """Properly close LMDB environment and free resources"""
        if hasattr(self, 'env') and self.env is not None:
            try:
                self.env.close()
            except Exception as e:
                print(f"⚠️  Warning: Error closing LMDB environment: {e}")
            finally:
                self.env = None

    def __del__(self):
        """Ensure LMDB environment is closed when object is destroyed"""
        self.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - properly close resources"""
        self.close()
