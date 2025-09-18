import os
import pickle
from typing import Iterator, List, Tuple

import lmdb
import numpy as np

from ..selfplay.selfplay import SelfPlayData


class DataBuffer:
    """LMDB-backed replay buffer with lazy augmentation for reduced memory overhead"""

    def __init__(
        self,
        db_path: str,
        max_size: int = 5_000_000,
        map_size: int = 64 * 1024**3,
        lazy_augmentation: bool = True,
    ):
        self.db_path = db_path
        self.max_size = max_size
        self.lazy_augmentation = lazy_augmentation
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
        """Add training examples with optional lazy augmentation"""
        if self.lazy_augmentation:
            # Store raw examples and augment on-demand
            data_to_store = data
        else:
            # Traditional approach: pre-generate all augmentations
            data_to_store = []
            for example in data:
                data_to_store.extend(self._augment_example(example))

        try:
            with self.env.begin(write=True) as txn:
                for example in data_to_store:
                    key = f"data_{self.write_idx}".encode()
                    if self.lazy_augmentation:
                        # Store with augmentation index for lazy loading
                        value = pickle.dumps((example, -1))  # -1 = original, 0-7 = augmentations
                    else:
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
                    for example in data_to_store:
                        key = f"data_{self.write_idx}".encode()
                        if self.lazy_augmentation:
                            value = pickle.dumps((example, -1))
                        else:
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
                    state=rot_state,
                    policy=rot_policy.flatten(),
                    value=example.value,
                    current_player=example.current_player,
                    last_move=example.last_move,
                    metadata=dict(example.metadata) if example.metadata else {},
                )
            )

            # Flip horizontally
            flip_state = np.flip(rot_state, axis=2)
            flip_policy = np.flip(rot_policy, axis=1)

            augmented.append(
                SelfPlayData(
                    state=flip_state,
                    policy=flip_policy.flatten(),
                    value=example.value,
                    current_player=example.current_player,
                    last_move=example.last_move,
                    metadata=dict(example.metadata) if example.metadata else {},
                )
            )

        return augmented

    def _apply_single_augmentation(self, example: SelfPlayData, aug_idx: int) -> SelfPlayData:
        """Apply a single augmentation transformation (0-7) to reduce memory usage"""
        state = example.state
        size = state.shape[-1]  # Board size from state shape
        policy = example.policy.reshape(size, size)

        # Apply specific augmentation
        rot_count = aug_idx // 2
        flip = bool(aug_idx % 2)

        # Rotate
        aug_state = np.rot90(state, rot_count, axes=(1, 2))
        aug_policy = np.rot90(policy, rot_count)

        # Flip if needed
        if flip:
            aug_state = np.flip(aug_state, axis=2)
            aug_policy = np.flip(aug_policy, axis=1)

        return SelfPlayData(
            state=aug_state,
            policy=aug_policy.flatten(),
            value=example.value,
            current_player=example.current_player,
            last_move=example.last_move,
            metadata=dict(example.metadata) if example.metadata else {},
        )

    def sample_batch(self, batch_size: int) -> List[SelfPlayData]:
        """Sample random batch from buffer with on-demand augmentation"""
        if self.size == 0:
            return []

        indices = np.random.randint(0, self.size, batch_size)
        batch = []

        with self.env.begin() as txn:
            for idx in indices:
                key = f"data_{idx}".encode()
                value = txn.get(key)
                if value:
                    loaded_data = pickle.loads(value)
                    if self.lazy_augmentation:
                        # Unpack stored data and apply random augmentation
                        if isinstance(loaded_data, tuple):
                            example, _ = loaded_data
                        else:
                            example = loaded_data
                        # Apply random augmentation (0-7) on-the-fly
                        aug_idx = np.random.randint(0, 8)
                        augmented_example = self._apply_single_augmentation(example, aug_idx)
                        batch.append(augmented_example)
                    else:
                        batch.append(loaded_data)

        return batch

    def sample(self, batch_size: int) -> List[SelfPlayData]:
        """Alias for sample_batch for backward compatibility"""
        return self.sample_batch(batch_size)

    def __len__(self) -> int:
        return self.size

    def get_stats(self) -> dict:
        """Get buffer statistics"""
        return {
            'size': self.size,
            'max_size': self.max_size,
            'write_idx': self.write_idx,
            'utilization': self.size / self.max_size if self.max_size > 0 else 0.0
        }

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
