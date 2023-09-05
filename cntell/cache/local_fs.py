import os
import pickle
import time
from cntell.common.cms import CacheManagementSystem
from cntell.common.component import Component, Input, Output


class LocalFilesystemCMS(CacheManagementSystem):
    def __init__(self, root_folder: str, max_records=64) -> None:
        super().__init__()
        
        self._root_folder = root_folder
        self._skip_cache = {}
        self._journal = {}
        self._max_records = 64
        
        
        
    def save_output(self, input: Input, output: Output) -> None:
        if len(self._skip_cache) == self._max_records:
            keys_to_delete = [k for k, _ in sorted(self._journal, key=lambda x: x[1])]
            for K in keys_to_delete:
                self._journal.pop(K)
                self._skip_cache.pop(K)
        
        input_checksum = input.checksum()
        self._skip_cache[input_checksum] = output
        self._journal[input_checksum] = output.timestamp
        
            
    def prepare(self) -> None:
        pass
        
        
    def can_skip(self, input: Input) -> bool:
        return input.checksum() in self._skip_cache

    
    def get_output(self, input: Input) -> Output:
        return self._skip_cache[input.checksum()]

    
    def suspend(self) -> None:
        with open(os.path.join(self._root_folder, 'skip_cache'), 'wb') as f:
            f.write(pickle.dumps(self._skip_cache))
            
        with open(os.path.join(self._root_folder, 'journal'), 'wb') as f:
            f.write(pickle.dumps(self._journal))

            
            
    @staticmethod
    def load(root_folder: str) -> 'LocalFilesystemCMS':
        cms = LocalFilesystemCMS(root_folder)
        cms.prepare()
        try:
            with open(os.path.join(root_folder, 'journal'), 'rb') as f:
                cms._journal = pickle.loads(f.read())
            
            with open(os.path.join(root_folder, 'skip_cache'), 'rb') as f:
                cms._skip_cache = pickle.loads(f.read())
        except: pass
        return cms
            
        