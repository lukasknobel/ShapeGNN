from torch import Tensor
from typing import Optional
from torch_sparse import SparseTensor

from torch import device as torch_device
class GraphData:
    def __init__(self, x: Tensor, adj_t: SparseTensor,
            pos: Tensor, y: Tensor, sub_x: Optional[Tensor], sub_adj_t: Optional[SparseTensor],
             batch: Tensor, sub_batch: Optional[Tensor], edge_index: Optional[Tensor],
             batch_lengths: Optional[Tensor], edge_batch: Optional[Tensor]):
        self.x = x
        self.adj_t = adj_t
        self.pos = pos
        self.y = y
        self.sub_x = sub_x
        self.sub_adj_t = sub_adj_t
        self.batch = batch
        self.sub_batch = sub_batch
        self.edge_index = edge_index
        self.batch_lengths = batch_lengths
        self.edge_batch = edge_batch

    def pin_memory(self):
        self.x = self.x.pin_memory()
        self.adj_t = self.adj_t.pin_memory()
        self.pos = self.pos.pin_memory()
        self.y = self.y.pin_memory()
        self.batch = self.batch.pin_memory()
        
        # These unneeded checks are so this can be Just In Timed (although JIT-ing the
        # whole model and data batching did not provide any performance benefits in initial
        # experiments)
        sub_x = self.sub_x
        if sub_x is not None:
            self.sub_x = sub_x.pin_memory()

        sub_adj_t = self.sub_adj_t
        if sub_adj_t is not None:
            self.sub_adj_t = sub_adj_t.pin_memory()

        sub_batch = self.sub_batch
        if sub_batch is not None:
            self.sub_batch = sub_batch.pin_memory()

        edge_index = self.edge_index
        if edge_index is not None:
            self.edge_index = edge_index.pin_memory()

        batch_lengths = self.batch_lengths
        if batch_lengths is not None:
            self.batch_lengths = batch_lengths.pin_memory()

        edge_batch = self.edge_batch
        if edge_batch is not None:
            self.edge_batch = edge_batch.pin_memory()

        return self

    def to(self, device: torch_device, non_blocking: bool):
        self.x = self.x.to(device,non_blocking=non_blocking)
        self.adj_t = self.adj_t.to_device(device,non_blocking=non_blocking)
        self.pos = self.pos.to(device,non_blocking=non_blocking)
        self.y = self.y.to(device,non_blocking=non_blocking)
        self.batch = self.batch.to(device,non_blocking=non_blocking)
        
        sub_x = self.sub_x
        if sub_x is not None:
            self.sub_x = sub_x.to(device,non_blocking=non_blocking)

        sub_adj_t = self.sub_adj_t
        if sub_adj_t is not None:
            self.sub_adj_t = sub_adj_t.to_device(device,non_blocking=non_blocking)
 
        sub_batch = self.sub_batch
        if sub_batch is not None:
            self.sub_batch = sub_batch.to(device, non_blocking=non_blocking)

        edge_index = self.edge_index
        if edge_index is not None:
            self.edge_index = edge_index.to(device, non_blocking=non_blocking)

        batch_lengths = self.batch_lengths
        if batch_lengths is not None:
            self.batch_lengths = batch_lengths.to(device, non_blocking=non_blocking)

        edge_batch = self.edge_batch
        if edge_batch is not None:
            self.edge_batch = edge_batch.to(device, non_blocking=non_blocking)

        return self
