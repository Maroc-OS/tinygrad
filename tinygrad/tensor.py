# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from __future__ import annotations
import time, math, itertools, functools
from contextlib import contextmanager
from typing import List, Tuple, Callable, Optional, ClassVar, Type, Union, Sequence, Dict, DefaultDict, cast, get_args, Set
from collections import defaultdict
import numpy as np

from tinygrad.dtype import DType, dtypes, ImageDType, ConstType, least_upper_float, least_upper_dtype, sum_acc_dtype
from tinygrad.helpers import argfix, make_pair, flatten, prod, all_int, round_up, merge_dicts, fully_flatten, argsort, getenv
from tinygrad.helpers import IMAGE, DEBUG, WINO, THREEFRY
from tinygrad.lazy import LazyBuffer
from tinygrad.multi import MultiLazyBuffer
from tinygrad.ops import LoadOps
from tinygrad.device import Device, Buffer, BufferOptions
from tinygrad.shape.symbolic import sint, Variable, MulNode, Node
from tinygrad.engine.realize import run_schedule
from tinygrad.engine.schedule import ScheduleItem, create_schedule_with_vars, memory_planner

# **** start with two base classes, Tensor and Function ****

class Function:
  def __init__(self, device:Union[str, Tuple[str, ...]], *tensors:Tensor):
    self.device = device
    self.needs_input_grad = [t.requires_grad for t in tensors]
    self.requires_grad = True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False
    if self.requires_grad: self.parents = tensors

  def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
  def backward(self, *args, **kwargs): raise RuntimeError(f"backward not implemented for {type(self)}")

  @classmethod
  def apply(fxn:Type[Function], *x:Tensor, **kwargs) -> Tensor:
    ctx = fxn(x[0].device, *x)
    ret = Tensor.__new__(Tensor)
    ret.lazydata, ret.requires_grad, ret.grad = ctx.forward(*[t.lazydata for t in x], **kwargs), ctx.requires_grad, None
    # We should not be setting requires_grad on inference tensors outside inference mode
    if ctx.requires_grad and Tensor.inference_tensor:
      raise RuntimeError("Setting requires_grad=True on inference tensor outside InferenceMode is not allowed.")
    elif ctx.requires_grad and Tensor.is_grad_enabled: # used by autograd engine
      Tensor.is_inference_mode_enabled = False
      Tensor.inference_tensor = False
      ret._ctx = ctx
    else:
      ret._ctx = None

    return ret

import tinygrad.function as F

def _loadop(op, shape:Tuple[sint,...], dtype:DType, device:Union[str, Tuple[str, ...]], arg=None, src:Tuple[LazyBuffer, ...]=()):
  if isinstance(device, str): return LazyBuffer.loadop(op, shape, dtype, device, arg, src)
  return MultiLazyBuffer([LazyBuffer.loadop(op, shape, dtype, d, arg, src) for d in device], None)

def _fromcpu(x: np.ndarray) -> LazyBuffer:
  ret = LazyBuffer.loadop(LoadOps.EMPTY, x.shape, dtypes.from_np(x.dtype), "NPY")
  # fake realize
  ret.buffer.allocate(x)
  del ret.srcs
  return ret

def _get_winograd_matcols(mat, dims:int, shp:Tuple[sint, ...], device:Union[str, Tuple[str, ...]]) -> List[List[Tensor]]:
  return [[Tensor.cat(*[Tensor.full(shp[:dim] + (1,) + shp[dim+1:], float(m[k]), device=device) for m in mat], dim=dim)
           for k in range(len(mat[0]))] for dim in range(dims)]

# winograd conv 3 kernel f(4x4,3x3) see: http://arxiv.org/abs/1509.09308
def _apply_winograd_matrix(mat, t:Tensor, dims:int) -> Tensor:
  # multiply mat_1 @ mat_2 @ t with foldable constants, where mat_i acts on vector t along dimension i; roughly kron(mat, mat) @ t
  # due to realize-before-expand rule in lazy.py, we must operate in this order: reshape -> expand -> arithmetic
  t_ = t.reshape(t.shape[:dims] + (1,) * dims + t.shape[dims:]).expand(t.shape[:dims] + (len(mat),) * dims + t.shape[dims:])  # add output dims
  # precalculate mat columns for each dim; prod(itertools.product(matcols)) gives the columns of kron(mat, mat, ...)
  matcols = _get_winograd_matcols(mat, dims, t_.shape[dims:], t_.device)
  # multiply each element of t_ by the corresponding stacked column of kron(mat, mat), producing only one view for each element of t
  ret = sum(prod(col[idx] for col, idx in zip(matcols, mat_is)) * t_[mat_is] for mat_is in itertools.product(range(len(mat[0])), repeat=dims))
  assert isinstance(ret, Tensor), "sum didn't return a Tensor"
  return ret

def _pad_left(*shps:Tuple[sint, ...], v=1): return tuple((v,) * (max(len(i_) for i_ in shps) - len(i)) + i for i in shps)
def _broadcast_shape(*shps:Tuple[sint, ...]): return tuple(0 if any(sh_ == 0 for sh_ in sh) else max(sh) for sh in zip(*_pad_left(*shps)))

class Tensor:
  """
  A `Tensor` is a multi-dimensional matrix containing elements of a single data type.

  ```python exec="true" session="tensor"
  from tinygrad import Tensor, dtypes, nn
  import numpy as np
  import math
  np.set_printoptions(precision=4)
  ```
  """
  __slots__ = "lazydata", "requires_grad", "grad", "_ctx"
  __deletable__ = ('_ctx',)
  is_grad_enabled: ClassVar[bool] = True
  is_train_enabled: ClassVar[bool] = False
  is_inference_mode_enabled: ClassVar[bool] = False
  inference_tensor: ClassVar[bool] = False

  @classmethod
  @contextmanager
  def train(cls, mode=True):
    prev_mode = cls.is_train_enabled
    cls.is_train_enabled = mode
    cls.inference_tensor = False
    try:
      yield
    finally:
      cls.is_train_enabled = prev_mode
      cls.inference_tensor = False

  @classmethod
  @contextmanager
  def no_grad(cls, mode=True):
    prev_mode = cls.is_grad_enabled
    cls.is_grad_enabled = not mode
    cls.inference_tensor = False
    try:
      yield
    finally:
      cls.is_grad_enabled = prev_mode
      cls.inference_tensor = False

  @classmethod
  @contextmanager
  def inference_mode(cls, mode=True):
    # TODO: disable view tracking, versioning, etc. All things that are not needed in that mode should be disabled
    prev_mode = cls.is_inference_mode_enabled
    cls.is_inference_mode_enabled = mode
    cls.inference_tensor = False
    try:
      yield
    finally:
      cls.is_inference_mode_enabled = prev_mode
      cls.inference_tensor = mode

  def __init__(self, data:Union[None, ConstType, List, Tuple, LazyBuffer, np.ndarray, bytes, MultiLazyBuffer, Variable],
               device:Optional[Union[str, tuple, list]]=None, dtype:Optional[DType]=None, requires_grad:Optional[bool]=None):
    assert dtype is None or isinstance(dtype, DType), f"invalid dtype {dtype}"
    device = tuple(Device.canonicalize(x) for x in device) if isinstance(device, (tuple, list)) else Device.canonicalize(device)
    # tensors have gradients, buffers do not
    self.grad: Optional[Tensor] = None

    # NOTE: this can be in three states. False and None: no gradient, True: gradient
    # None (the default) will be updated to True if it's put in an optimizer
    self.requires_grad: Optional[bool] = requires_grad

    # internal variables used for autograd graph construction
    self._ctx: Optional[Function] = None
    if isinstance(data, LazyBuffer): assert dtype is None or dtype == data.dtype, "dtype doesn't match, and casting isn't supported"
    elif isinstance(data, get_args(ConstType)): data = _loadop(LoadOps.CONST, tuple(), dtype or dtypes.from_py(data), device, data)
    elif isinstance(data, Variable): data = _loadop(LoadOps.CONST, tuple(), dtype or dtypes.from_py(data.unbind()[1]), device, data)
    elif isinstance(data, bytes): data = _fromcpu(np.frombuffer(data, np.uint8))
    elif data is None: data = _loadop(LoadOps.EMPTY, (0,), dtype or dtypes.default_float, device)
    elif isinstance(data, list):
      if dtype is None:
        if (d := fully_flatten(data)) and all(isinstance(s, bool) for s in d): dtype = dtypes.bool
        else: dtype = dtypes.default_int if d and all_int(d) else dtypes.default_float
      if dtype == dtypes.bfloat16: data = Tensor(_fromcpu(np.array(data, np.float32)), device=device).cast(dtypes.bfloat16).lazydata
      else: data = _fromcpu(np.array(data, dtype.np))
    elif isinstance(data, np.ndarray):
      if data.shape == (): data = _loadop(LoadOps.CONST, tuple(), dtype or dtypes.from_np(data.dtype), device, data.item())
      else: data = _fromcpu(data.astype(dtype.np) if dtype is not None and dtype.np is not None else data)

    # data is a LazyBuffer, but it might be on the wrong device
    if not isinstance(data, (LazyBuffer, MultiLazyBuffer)): raise RuntimeError(f"can't create Tensor from {data!r} with type {type(data)}")
    if isinstance(device, tuple):
      # TODO: what if it's a MultiLazyBuffer on other devices?
      self.lazydata: Union[LazyBuffer, MultiLazyBuffer] = MultiLazyBuffer.from_sharded(data, device, None) if isinstance(data, LazyBuffer) else data
    else:
      self.lazydata = data if data.device == device else data.copy_to_device(device)

  def __repr__(self): return f"<Tensor {self.lazydata!r} on {self.device} with grad {(self.grad.lazydata if self.grad is not None else None)!r}>"

  # Python has a non moving GC, so this should be okay
  def __hash__(self): return id(self)

  def __bool__(self): raise TypeError("__bool__ on Tensor is not defined")

  def __len__(self): return self.shape[0] if len(self.shape) else 1

  @property
  def device(self) -> Union[str, Tuple[str, ...]]: return self.lazydata.device

  @property
  def shape(self) -> Tuple[sint, ...]: return self.lazydata.shape

  @property
  def dtype(self) -> DType: return self.lazydata.dtype

  # ***** data handlers ****

  def schedule_with_vars(self, *lst:Tensor, seen:Optional[Set[LazyBuffer]]=None) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
    """Create the schedule needed to realize these Tensor(s), with Variables."""
    if getenv("FUZZ_SCHEDULE"):
      from test.external.fuzz_schedule import fuzz_schedule
      fuzz_schedule(flatten([x.lazydata.lbs for x in (self,)+lst]))
    schedule, var_vals = create_schedule_with_vars(flatten([x.lazydata.lbs for x in (self,)+lst]), seen)
    return memory_planner(schedule), var_vals

  def schedule(self, *lst:Tensor, seen:Optional[Set[LazyBuffer]]=None) -> List[ScheduleItem]:
    """Create the schedule needed to realize these Tensor(s)."""
    schedule, var_vals = self.schedule_with_vars(*lst, seen=seen)
    assert len(var_vals) == 0
    return schedule

  def realize(self, *lst:Tensor, do_update_stats=True) -> Tensor:
    """Trigger the computation needed to create these Tensor(s)."""
    run_schedule(*self.schedule_with_vars(*lst), do_update_stats=do_update_stats)
    return self

  def replace(self, x:Tensor) -> Tensor:
    """
    Replace the data of this tensor with the data of another tensor. Only the shape of the tensors must match.
    """
    # used for replacing a Tensor with a new version of it (potentially with a different device and dtype)
    assert not x.requires_grad and getattr(self, '_ctx', None) is None
    assert self.shape == x.shape, f"replace shape mismatch {self.shape} != {x.shape}"
    self.lazydata = x.lazydata
    return self

  def assign(self, x) -> Tensor:
    # TODO: this is a hack for writing to DISK. remove with working assign
    if isinstance(self.device, str) and self.device.startswith("DISK"):
      if x.__class__ is not Tensor: x = Tensor(x, device="NPY", dtype=self.dtype)
      self.contiguous().realize().lazydata.base.realized.copyin(x.numpy().data)
      return self
    if x.__class__ is not Tensor: x = Tensor(x, device=self.device, dtype=self.dtype)
    if DEBUG >= 4: print(f"assign {self.lazydata} <- {x.lazydata}")
    if self.lazydata is x.lazydata: return self  # a self assign is a NOOP
    # NOTE: we allow cross device assign
    assert self.shape == x.shape, f"assign shape mismatch {self.shape} != {x.shape}"
    assert self.device == x.device, f"assign device mismatch {self.device} != {x.device}"
    assert self.dtype == x.dtype, f"assign dtype mismatch {self.dtype} != {x.dtype}"
    assert not isinstance(self.lazydata, MultiLazyBuffer) or self.lazydata.axis == x.lazydata.axis, "axis must match on MultiLazyBuffer"
    assert not x.requires_grad  # self requires_grad is okay?
    if not self.lazydata.is_realized(): return self.replace(x)
    self.lazydata = self.lazydata.assign(x.lazydata)
    return self
  def detach(self) -> Tensor:
    """
    Returns a new tensor with the same data as this tensor, but detached from the autograd graph.
    """
    return Tensor(self.lazydata, device=self.device, requires_grad=False)

  def _data(self) -> memoryview:
    if 0 in self.shape: return memoryview(bytearray(0))
    # NOTE: this realizes on the object from as_buffer being a Python object
    cpu = self.cast(self.dtype.scalar()).contiguous().to("CLANG").realize()
    buf = cast(Buffer, cast(LazyBuffer, cpu.lazydata).base.realized)
    if self.device != "CLANG": buf.options = BufferOptions(nolru=True)
    return buf.as_buffer(allow_zero_copy=True if self.device != "CLANG" else False)

  def data(self) -> memoryview:
    """
    Returns the data of this tensor as a memoryview.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(np.frombuffer(t.data(), dtype=np.int32))
    ```
    """
    assert self.dtype.fmt is not None, f"no fmt dtype for {self.dtype}"
    assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
    return self._data().cast(self.dtype.fmt, self.shape)
  def item(self) -> ConstType:
    """
    Returns the value of this tensor as a standard Python number.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor(42)
    print(t.item())
    ```
    """
    assert self.dtype.fmt is not None, f"no fmt dtype for {self.dtype}"
    assert self.numel() == 1, "must have one element for item"
    return self._data().cast(self.dtype.fmt)[0]
  # TODO: should be Tensor.tolist() -> Union[List[ConstType], ConstType]. The List is Sequence because mypy expects memoryview.tolist() -> list[int]
  # src: https://github.com/python/mypy/blob/release-1.6/mypy/typeshed/stdlib/builtins.pyi#L803
  def tolist(self) -> Union[Sequence[ConstType], ConstType]:
    """
    Returns the value of this tensor as a nested list.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(t.tolist())
    ```
    """
    return self.data().tolist()
  def numpy(self) -> np.ndarray:
    """
    Returns the value of this tensor as a numpy array.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(t.numpy())
    ```
    """
    if self.dtype == dtypes.bfloat16: return self.float().numpy()
    assert self.dtype.np is not None, f"no np dtype for {self.dtype}"
    assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
    return np.frombuffer(self._data(), dtype=self.dtype.np).reshape(self.shape)

  def to(self, device:Optional[Union[str, Tuple[str, ...]]]) -> Tensor:
    """
    Moves the tensor to the given device.
    """
    device = tuple(Device.canonicalize(x) for x in device) if isinstance(device, (tuple, list)) else Device.canonicalize(device)
    if device == self.device: return self
    if not isinstance(device, str): return self.shard(device)
    ret = Tensor(self.lazydata, device, requires_grad=self.requires_grad)
    if self.grad is not None: ret.grad = self.grad.to(device)
    if hasattr(self, '_ctx'): ret._ctx = self._ctx
    return ret

  def to_(self, device:Optional[Union[str, Tuple[str, ...]]]):
    """
    Moves the tensor to the given device in place.
    """
    real = self.to(device)
    # TODO: is this assign?
    if self.grad is not None and real.grad is not None: self.grad.lazydata = real.grad.lazydata
    self.lazydata = real.lazydata

  def shard(self, devices:Tuple[str, ...], axis:Optional[int]=None) -> Tensor:
    """
    Shards the tensor across the given devices.
    """
    assert isinstance(self.lazydata, LazyBuffer), "can't shard a MultiLazyBuffer"
    canonical_devices = tuple(Device.canonicalize(x) for x in devices)
    if axis is not None and axis < 0: axis += len(self.shape)
    return Tensor(MultiLazyBuffer.from_sharded(self.lazydata, canonical_devices, axis), device=canonical_devices, requires_grad=self.requires_grad)

  def shard_(self, devices:Tuple[str, ...], axis:Optional[int]=None):
    """
    Shards the tensor across the given devices in place.
    """
    self.lazydata = self.shard(devices, axis).lazydata
    return self

  @staticmethod
  def from_node(y:Node, **kwargs) -> Tensor:
    if isinstance(y, MulNode): return Tensor.from_node(y.a, **kwargs) * y.b
    if isinstance(y, Variable): return Tensor(y, **kwargs, requires_grad=False)
    raise RuntimeError(f"unhandled Node {y}")

  # ***** creation llop entrypoint *****

  @staticmethod
  def _loadop(op, shape, device:Optional[Union[Tuple[str, ...], str]]=None, dtype:Optional[DType]=None, arg=None, **kwargs):
    if isinstance(device, tuple):
      return Tensor(MultiLazyBuffer([LazyBuffer.loadop(op, shape, dtype or dtypes.default_float, Device.canonicalize(d), arg) \
                                      for d in device], None), device, dtype, **kwargs)
    return Tensor(LazyBuffer.loadop(op, shape, dtype or dtypes.default_float, Device.canonicalize(device), arg), device, dtype, **kwargs)

  @staticmethod
  def empty(*shape, **kwargs):
    """
    Creates an empty tensor with the given shape.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.empty(2, 3)
    print(t.shape)
    ```
    """
    return Tensor._loadop(LoadOps.EMPTY, argfix(*shape), **kwargs)

  _seed: int = int(time.time())
  _rng_counter: Optional[Tensor] = None
  @staticmethod
  def manual_seed(seed=0):
    """
    Sets the seed for random operations.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor._seed)
    ```
    """
    Tensor._seed, Tensor._rng_counter = seed, Tensor([0], dtype=dtypes.uint32, requires_grad=False)

  @staticmethod
  def rand(*shape, device:Optional[Union[Tuple[str, ...], str]]=None, dtype:Optional[DType]=None, **kwargs):
    """
    Creates a tensor with the given shape, filled with random values between the interval `[0, 1)`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.rand(2, 3)
    print(t.numpy())
    ```
    """
    if Tensor._rng_counter is None: Tensor._rng_counter = Tensor([0], dtype=dtypes.uint32, requires_grad=False)
    if not THREEFRY.value:
      # for bfloat16, numpy rand passes buffer in float
      if (dtype or dtypes.default_float) == dtypes.bfloat16:
        return Tensor.rand(*shape, **kwargs, device=device, dtype=dtypes.float).cast(dtypes.bfloat16)
      return Tensor._loadop(LoadOps.CUSTOM, argfix(*shape), arg=custom_random, device=device, dtype=dtype, **kwargs)

    # threefry
    if (num := prod((shape:=argfix(*shape)))) == 0: return Tensor.zeros(shape, device=device, dtype=dtype, **kwargs)
    counts1 = (Tensor.arange(math.ceil(num / 2), device=device, dtype=dtypes.uint32, requires_grad=False)+Tensor._rng_counter.to(device)).realize()
    counts2 = counts1 + math.ceil(num / 2)
    Tensor._rng_counter.assign(Tensor._rng_counter + num).realize()

    rotations = [[13, 15, 26, 6], [17, 29, 16, 24]]
    ks = [0x0, Tensor._seed ^ 0x0 ^ 0x1BD11BDA, Tensor._seed]

    x = [counts1 + ks[-1], counts2 + ks[0]]
    for i in range(5):
      for r in rotations[i % 2]: x[0], x[1] = (x0 := x[0] + x[1]), x0 ^ ((x[1] << r) + (x[1] >> (32 - r)))
      x = [(x[0] + ks[i % 3]), (x[1] + ks[(i + 1) % 3] + i + 1)]
    out = x[0].cat(x[1]).rshift(8).cast(dtypes.float32).div(2 ** 24)[:num]
    out = out.reshape(shape).cast(dtypes.default_float if dtype is None else dtype)
    out.requires_grad = kwargs.get("requires_grad")
    return out.contiguous()

  # ***** creation helper functions *****

  @staticmethod
  def full(shape:Tuple[sint, ...], fill_value:ConstType, **kwargs):
    """
    Creates a tensor with the given shape, filled with the given value.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.full((2, 3), 42).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.full((2, 3), False).numpy())
    ```
    """
    return Tensor(fill_value, **kwargs).reshape((1, )*len(new_shape := argfix(shape))).expand(new_shape)

  @staticmethod
  def zeros(*shape, **kwargs):
    """
    Creates a tensor with the given shape, filled with zeros.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.zeros(2, 3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.zeros(2, 3, dtype=dtypes.int32).numpy())
    ```
    """
    return Tensor.full(argfix(*shape), 0.0, **kwargs)

  @staticmethod
  def ones(*shape, **kwargs):
    """
    Creates a tensor with the given shape, filled with ones.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(2, 3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(2, 3, dtype=dtypes.int32).numpy())
    ```
    """
    return Tensor.full(argfix(*shape), 1.0, **kwargs)

  @staticmethod
  def arange(start, stop=None, step=1, **kwargs):
    """
    If `stop` is not specified, creates a tensor with the given shape, filled with values from `0` to `start` with the given step size.

    If `stop` is specified, creates a tensor with the given shape, filled with values from `start` to `stop` with the given step size.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5, 10).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5, 10, 2).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.arange(5.5, 10, 2).numpy())
    ```
    """
    if stop is None: stop, start = start, 0
    assert all(isinstance(s, (int, float)) for s in (start, stop, step)), f"symbolic arange not supported {start=}, {stop=}, {step=}"
    dtype = kwargs.pop("dtype", dtypes.default_float if any(isinstance(x, float) for x in (start, stop, step)) else dtypes.default_int)
    return (Tensor.full((math.ceil((stop-start)/step),), step, dtype=dtype, **kwargs)._cumsum() + (start - step)).cast(dtype)

  @staticmethod
  def eye(dim:int, **kwargs):
    """
    Creates an identity matrix of the given dimension.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.eye(3).numpy())
    ```
    """
    return Tensor.ones((dim,1),**kwargs).pad((None,(0,dim))).flatten().shrink(((0,dim*dim),)).reshape(dim, dim)

  def full_like(self, fill_value:ConstType, **kwargs):
    """
    Creates a tensor with the same shape as `tensor`, filled with the given value.
    If `dtype` is not specified, the dtype of `tensor` is used.

    You can pass in the `device` keyword argument to control device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(Tensor.full_like(t, 42).numpy())
    ```
    """
    return Tensor.full(self.shape, fill_value, dtype=kwargs.pop("dtype", self.dtype), device=kwargs.pop("device", self.device), **kwargs)

  def zeros_like(self, **kwargs):
    """
    Creates a tensor with the same shape as `tensor`, filled with zeros.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(Tensor.zeros_like(t).numpy())
    ```
    """
    return self.full_like(0, **kwargs)

  def ones_like(self, **kwargs):
    """
    Creates a tensor with the same shape as `tensor`, filled with ones.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.zeros(2, 3)
    print(Tensor.ones_like(t).numpy())
    ```
    """
    return self.full_like(1, **kwargs)

  # ***** rng hlops *****

  @staticmethod
  def randn(*shape, dtype:Optional[DType]=None, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a normal distribution with mean `0` and standard deviation `1`.
    If `dtype` is not specified, the default type is used.

    You can pass in the `device` keyword argument to control device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.randn(2, 3).numpy())
    ```
    """
    # https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    src = Tensor.rand((2, *argfix(*shape)), **{**kwargs, "dtype": dtypes.float32})
    return src[0].mul(2*math.pi).cos().mul((1 - src[1]).log().mul(-2).sqrt()).cast(dtype or dtypes.default_float)

  @staticmethod
  def randint(*shape, low=0, high=10, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random integer values from the interval `[low, high)`.
    If `dtype` is not specified, the default type is used.

    You can pass in the `device` keyword argument to control device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.randint(2, 3, low=5, high=10).numpy())
    """
    assert dtypes.is_int(dtype := kwargs.pop("dtype", dtypes.int32)), f"Unsupported dtype {dtype} for randint"
    return Tensor.uniform(*shape, low=low, high=high, dtype=dtype, **kwargs)

  @staticmethod
  def normal(*shape, mean=0.0, std=1.0, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a normal distribution with the given mean and standard deviation.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.normal(2, 3, mean=10, std=2).numpy())
    ```
    """
    return (std * Tensor.randn(*shape, **kwargs)) + mean

  @staticmethod
  def uniform(*shape, low=0.0, high=1.0, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values from a uniform distribution with the given lower and upper bounds.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.uniform(2, 3, low=2, high=10).numpy())
    ```
    """
    dtype = kwargs.pop("dtype", dtypes.default_float)
    return ((high-low) * Tensor.rand(*shape, **kwargs)).cast(dtype) + low

  @staticmethod
  def scaled_uniform(*shape, **kwargs) -> Tensor:
    """
    Creates a tensor with the given shape, filled with random values
    from a uniform distribution with a mean of zero and a standard deviation of `(prod(shape)**-0.5`.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.scaled_uniform(2, 3).numpy())
    ```
    """
    return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul(prod(argfix(*shape))**-0.5)

  # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
  @staticmethod
  def glorot_uniform(*shape, **kwargs) -> Tensor:
    """
    <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform>

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.glorot_uniform(2, 3).numpy())
    ```
    """
    return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul((6/(argfix(*shape)[0]+prod(argfix(*shape)[1:])))**0.5)

  # https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_
  @staticmethod
  def kaiming_uniform(*shape, a:float = 0.01, **kwargs) -> Tensor:
    """
    <https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_>

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.kaiming_uniform(2, 3).numpy())
    ```
    """
    bound = math.sqrt(3.0) * math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(argfix(*shape)[1:]))
    return Tensor.uniform(*shape, low=-bound, high=bound, **kwargs)

  # https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_
  @staticmethod
  def kaiming_normal(*shape, a:float = 0.01, **kwargs) -> Tensor:
    """
    <https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_>

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    print(Tensor.kaiming_normal(2, 3).numpy())
    ```
    """
    std = math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(argfix(*shape)[1:]))
    return Tensor.normal(*shape, mean=0.0, std=std, **kwargs)

  def multinomial(self:Tensor, num_samples:int = 1, replacement:bool = False) -> Tensor:
    assert 1 <= self.ndim <= 2 and num_samples > 0, f"{self.ndim=} must be 1 or 2 dim, {num_samples=} must be positive"
    assert replacement or num_samples == 1, "no replacement only supports num_samples = 1"
    weight = self.unsqueeze(0) if self.ndim == 1 else self
    cdf = (cw := weight.cumsum(1).float()) / cw[:, -1].unsqueeze(1)
    unif_samples = Tensor.rand(num_samples, cdf.shape[0], 1, device=self.device)
    indices = (unif_samples.expand((-1, -1, cdf.shape[1])) >= cdf).sum(2).permute((1, 0))
    return (indices.squeeze(0) if self.ndim == 1 else indices).cast(dtypes.int32)

  # ***** toposort and backward pass *****

  def _deepwalk(self):
    def _walk(node, visited):
      visited.add(node)
      if getattr(node, "_ctx", None):
        for i in node._ctx.parents:
          if i not in visited: yield from _walk(i, visited)
        yield node
    return list(_walk(self, set()))

  def backward(self) -> Tensor:
    """
    Propagates the gradient of a tensor backwards through the computation graph.
    Must be used on a scalar tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    t.sum().backward()
    print(t.grad.numpy())
    ```
    """
    assert self.shape == tuple(), f"backward can only be called for scalar tensors, but it has shape {self.shape})"

    # fill in the first grad with one. don't use Tensor.ones because we don't need contiguous
    # this is "implicit gradient creation"
    self.grad = Tensor(1.0, dtype=self.dtype, device=self.device, requires_grad=False)

    for t0 in reversed(self._deepwalk()):
      if t0.grad is None: raise RuntimeError(f"tensor {t0} has no grad")
      grads = t0._ctx.backward(t0.grad.lazydata)
      grads = [Tensor(g, device=self.device, requires_grad=False) if g is not None else None
        for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]
      for t, g in zip(t0._ctx.parents, grads):
        if g is not None and t.requires_grad:
          assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
          t.grad = g if t.grad is None else (t.grad + g)
      del t0._ctx
    return self

  # ***** movement low level ops *****

  def view(self, *shape) -> Tensor:
    """`.view` is an alias for `.reshape`."""
    return self.reshape(shape)

  def reshape(self, shape, *args) -> Tensor:
    """
    Returns a tensor with the same data as the original tensor but with a different shape.
    `shape` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6)
    print(t.reshape(2, 3).numpy())
    ```
    """
    new_shape = argfix(shape, *args)
    new_shape = tuple([-prod(self.shape) // prod(new_shape) if s == -1 else (s if s is not None else self.shape[i]) for i,s in enumerate(new_shape)])
    return F.Reshape.apply(self, shape=new_shape) if new_shape != self.shape else self

  def expand(self, shape, *args) -> Tensor:
    """
    Returns a tensor that is expanded to the shape that is specified.
    Expand can also increase the number of dimensions that a tensor has.

    Passing a `-1` or `None` to a dimension means that its size will not be changed.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3])
    print(t.expand(4, -1).numpy())
    ```
    """
    return self._broadcast_to(tuple(sh if s==-1 or s is None else s for s, sh in zip(*(_pad_left(argfix(shape, *args), self.shape)))))

  def permute(self, order, *args) -> Tensor:
    """
    Returns a tensor that is a permutation of the original tensor.
    The new tensor has the same data as the original tensor but with the dimensions permuted according to the order specified.
    `order` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.permute(1, 0).numpy())
    ```
    """
    return F.Permute.apply(self, order=argfix(order, *args))

  def flip(self, axis, *args) -> Tensor:
    """
    Returns a tensor that reverses the order of the original tensor along given axis.
    `axis` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.flip(0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.flip((0, 1)).numpy())
    ```
    """
    return F.Flip.apply(self, axis=[x if x >= 0 else x+len(self.shape) for x in argfix(axis, *args)])

  def shrink(self, arg:Tuple[Optional[Tuple[sint, sint]], ...]) -> Tensor:
    """
    Returns a tensor that shrinks the each axis based on input arg.
    `arg` has the same length as `self.ndim`.
    For each axis, it can be `None`, which means no shrink, or a tuple `(start, end)` that works the same as python slice.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(3, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.shrink(((None, (1, 3)))).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.shrink((((0, 2), (0, 2)))).numpy())
    ```
    """
    if all(x is None or x == (0,s) for x,s in zip(arg, self.shape)): return self
    return F.Shrink.apply(self, arg=tuple(x if x is not None else (0,s) for x,s in zip(arg, self.shape)))

  def pad(self, arg:Tuple[Optional[Tuple[sint, sint]], ...], value:float=0.0) -> Tensor:
    """
    Returns a tensor that pads the each axis based on input arg.
    arg has the same length as `self.ndim`.
    For each axis, it can be `None`, which means no pad, or a tuple `(pad_before, pad_after)`.
    If `value` is specified, the tensor is padded with `value` instead of `0.0`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.pad(((None, (1, 2)))).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.pad(((None, (1, 2))), -2).numpy())
    ```
    """
    if all(x is None or x == (0,0) for x in arg): return self
    ret = F.Pad.apply(self, arg=(narg:=tuple(x if x is not None else (0,0) for x in arg)))
    return ret if 0 == value else ret + F.Pad.apply(Tensor.ones_like(self), arg=narg).where(0, value)

  # ***** movement high level ops *****

  # Supported Indexing Implementations:
  #   1. Int indexing (no copy)
  #     - for all dims where there's int, shrink -> reshape
  #     - negative indices are taken relative to the end of the sequence, so X[-2] returns the 2nd-to-last element
  #     - X = Tensor.rand(4,5,9); X[2,-2] shrinks the Tensor to X.shrink(((2, 3), (3, 4), (0, 9))) -> X.shape=(1,1,9)
  #     - Then we reshape (collapse) the int dim away such that for X: (1,1,9) -> (9,)
  #   2. Slice indexing (no copy)
  #     - for all dims where slice is start:end:stride, shrink -> Optional[flip] -> pad -> reshape -> shrink
  #     - first shrink the Tensor to X.shrink(((start, end),))
  #     - then we apply stride through Optional[flip] -> pad -> reshape -> shrink
  #       - flip where dim value is negative
  #       - pad 0's on dims such that reshaping [dim_size_padded] -> [dim_size_padded // stride, stride] is possible
  #       - shrink [dim_size_padded // stride, stride] -> [dim_size_padded // stride, 1]
  #       - reshape [dim_size_padded // stride, 1] -> [dim_size_padded // stride] and now you have your stride
  #   3. None indexing (no copy)
  #     - reshape (inject) a dim at the dim where there's None
  #   4. Tensor indexing (copy)
  #     - use Tensor.arange == tensor_index to create masks for dims with Tensors (adds a dim for each mask)
  #     - combine masks together with mul
  #     - apply mask to self by mask * self
  #     - sum reduce away the extra dims added from creating masks
  # Tiny Things:
  #   1. Supported indices: Union[int, slice, Tensor, None, List, Tuple, Ellipsis]
  #     - for any list, List[Union[List, Tuple, int]], must have homogeneous shape
  #     - for any tuple, Tuple[Union[List, Tuple, int]], must have homogeneous shape
  #   2. Bool indexing is not supported
  #   3. Out of bounds Tensor indexing results in 0
  #     - e.g: Tensor([1, 2, 3])[Tensor([4, 3, 2])] -> [0, 0, 3] index 4 and 3 are OOB
  def __getitem__(self, indices) -> Tensor:
    # 1. indices normalization and validation
    # treat internal tuples and lists as Tensors and standardize indices to list type
    if isinstance(indices, list) and all_int(indices): indices = [Tensor(indices, self.device, requires_grad=False)]
    elif isinstance(indices, (tuple, list)):
      indices = [Tensor(list(i), self.device, requires_grad=False) if isinstance(i, (tuple, list)) else i for i in indices]
    else: indices = [indices]

    # turn scalar Tensors into const val for int indexing if possible
    indices = [self._to_const_val(i) if isinstance(i, Tensor) and i.shape == () else i for i in indices]
    # move Tensor indices to the same device as self
    indices = [i.to(self.device) if isinstance(i, Tensor) else i for i in indices]

    # filter ellipsis and fill with slice(None) or fill rest of indices with slice(None)
    ellipsis_idx = [dim for dim, i in enumerate(indices) if i is Ellipsis]
    fill_idx = ellipsis_idx[0] if ellipsis_idx else len(indices)
    num_indices = len(indices) - len(ellipsis_idx) - sum(1 for i in indices if i is None)
    indices[fill_idx:fill_idx+1] = [slice(None)] * (len(self.shape) - num_indices)

    # use Dict[type, List[dimension]] to track elements in indices
    type_dim: DefaultDict[Union[type, None], List[int]] = defaultdict(list)

    # record None for dimension injection later and filter None and record rest of indices
    type_dim[None] = [dim for dim, i in enumerate(indices) if i is None]
    indices_filtered = [v for v in indices if v is not None]
    for dim,i in enumerate(indices_filtered): type_dim[type(i)].append(dim)

    for index_type in type_dim:
      if index_type not in [None, int, slice, Tensor]: raise IndexError(f"{index_type=} not supported")
    if len(ellipsis_idx) > 1: raise IndexError("indices can only have a single ellipsis ('...')")
    if num_indices > self.ndim: raise IndexError(f"too many {num_indices=} for {self.ndim=}")

    # 2. basic indexing, uses only movement ops (no copy)
    # currently indices_filtered: Tuple[Union[slice, int, Tensor], ...]
    # turn indices in indices_filtered to Tuple[shrink_arg, strides]
    for dim in type_dim[int]:
      if (index := indices_filtered[dim]) >= (size := self.shape[dim]) or index < -size:
        raise IndexError(f"{index=} is out of bounds on {dim=} with {size=}")
      indices_filtered[dim] = ((index, index+1), 1) if index >= 0 else ((size+index, size+index+1), 1)
    for dim in type_dim[slice]:
      if (index := indices_filtered[dim]).step == 0: raise ValueError(f"{index=} on {dim=} cannot have 0 as step")
      s, e, st = index.indices(self.shape[dim])
      indices_filtered[dim] = ((0, 0) if (st * (e - s)) < 0 else (s, e) if st > 0 else (e+1, s+1), st)
    # record tensors and skip all Tensor dims for basic indexing
    tensor_index: List[Tensor] = []
    for dim in type_dim[Tensor]:
      tensor_index.append(index := indices_filtered[dim])
      if not dtypes.is_int(index.dtype): raise IndexError(f"{index.dtype=} on {dim=} is not supported, only int tensor indexing is supported")
      indices_filtered[dim] = ((0, self.shape[dim]), 1)

    new_slice, strides = ((),()) if not indices_filtered else zip(*indices_filtered)
    ret = self.shrink(new_slice).flip(tuple(i for i, s in enumerate(strides) if s < 0))
    if any(abs(s) != 1 for s in strides):
      strides = tuple(abs(s) for s in strides)
      ret = ret.pad(tuple((0, round_up(sh, s) - sh) for s, sh in zip(strides, ret.shape)))
      ret = ret.reshape(tuple(flatten((sh // s, s) for s, sh in zip(strides, ret.shape))))
      ret = ret.shrink(tuple(flatten(((0, sh), (0, 1)) for sh in ret.shape[::2]))).reshape(ret.shape[::2])

    # inject 1 for dim where it's None and collapse dim for int
    new_shape = list(ret.shape)
    for dim in type_dim[None]: new_shape.insert(dim, 1)
    for dim in (dims_collapsed := tuple(dim + sum(1 for d in type_dim[None] if dim >= d) for dim in reversed(type_dim[int]))): new_shape.pop(dim)

    ret = ret.reshape(new_shape)
    assert all_int(ret.shape), f"does not support symbolic shape {ret.shape}"

    # 3. advanced indexing (copy)
    if type_dim[Tensor]:
      # calculate dim of current ret by subtracting dims collapsed and adding dims injected up until tensor_dim
      def calc_dim(tensor_dim:int) -> int:
        return tensor_dim - sum(1 for d in dims_collapsed if tensor_dim >= d) + sum(1 for d in type_dim[None] if tensor_dim >= d)

      # track tensor_dim and tensor_index using a dict
      # calc_dim to get dim and use that to normalize the negative tensor indices
      idx: Dict[int,Tensor] = {(dim := calc_dim(td)):(tensor<0).where(ret.shape[dim],0) + tensor for td,tensor in zip(type_dim[Tensor], tensor_index)}

      masks, first_dim, last_dim = [], min(idx.keys()), max(idx.keys())
      pre_reduce_shape = ret.shape[:first_dim] + (big_shape := _broadcast_shape(*(t.shape for t in idx.values()))) + ret.shape[first_dim:]

      # create masks
      for dim, i in idx.items():
        try: i = i.reshape(i.shape + (1,)*(ret.ndim - first_dim)).expand(pre_reduce_shape)
        except ValueError as e: raise IndexError(f"cannot broadcast indices: {e}") from e
        a = Tensor.arange(ret.shape[dim], device=self.device, requires_grad=False).reshape((ret.shape[dim],) + (1,)*(ret.ndim - dim - 1))
        masks.append(i == a)

      # reduce masks to 1 mask
      mask: Tensor = functools.reduce(lambda x,y: x.mul(y), masks)

      # inject 1's for the extra dims added in create masks
      sh = ret.shape[:first_dim] + (1,) * len(big_shape) + ret.shape[first_dim:]
      # sum reduce the extra dims introduced in create masks
      ret = (ret.reshape(sh) * mask).sum(tuple(i + len(big_shape) for i in idx.keys()), acc_dtype=ret.dtype)

      # special permute case
      if first_dim != 0 and len(idx) != 1 and tuple(idx.keys()) != tuple(range(first_dim, last_dim+1)):
        ret = ret.permute(*range(first_dim, first_dim+len(big_shape)), *range(0, first_dim), *range(first_dim+len(big_shape), ret.ndim))
    return ret

  def __setitem__(self, indices, v:Union[Tensor, ConstType]) -> None:
    if isinstance(self.device, str) and self.device.startswith("DISK"):
      self.__getitem__(indices).assign(v)
      return
    # NOTE: check that setitem target is valid first
    assert all(lb.st.contiguous for lb in self.lazydata.lbs), "setitem target needs to be contiguous"
    if not isinstance(v, (Tensor, float, int, bool)): raise TypeError(f"can't set a {type(v).__name__} to a Tensor")
    if not isinstance(v, Tensor): v = Tensor(v, device=self.device, dtype=self.dtype)
    if self.requires_grad or v.requires_grad: raise NotImplementedError("setitem with requires_grad is not supported")
    if isinstance(indices, (Tensor, list)) or (isinstance(indices, tuple) and any(isinstance(i, (Tensor, list)) for i in indices)):
      raise NotImplementedError("Advanced indexing setitem is not currently supported")

    assign_to = self.realize().__getitem__(indices)
    # NOTE: contiguous to prevent const folding.
    v = v.cast(assign_to.dtype)._broadcast_to(_broadcast_shape(assign_to.shape, v.shape)).contiguous()
    assign_to.assign(v).realize()

  # NOTE: using _slice is discouraged and things should migrate to pad and shrink
  def _slice(self, arg:Sequence[Optional[Tuple[int, sint]]], value:float=0) -> Tensor:
    arg_ = tuple(a if a is not None else (0, s) for s,a in zip(self.shape, arg))
    padding = tuple((max(0, -l), max(0, r-s)) for s,(l,r) in zip(self.shape, arg_))
    return self.pad(padding, value=value).shrink(tuple((l + pl, r + pl) for (l,r),(pl,_) in zip(arg_, padding)))

  def gather(self:Tensor, dim:int, index:Tensor) -> Tensor:
    """
    Gathers values along an axis specified by `dim`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2], [3, 4]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.gather(1, Tensor([[0, 0], [1, 0]])).numpy())
    ```
    """
    assert index.ndim == self.ndim, f"self.ndim must equal index.ndim, {self.ndim=}, {index.ndim=}"
    assert all(s >= i for s,i in zip(self.shape, index.shape)), "all dim of index.shape must be smaller than self.shape"
    dim = self._resolve_dim(dim)
    index = index.to(self.device).transpose(0, dim).unsqueeze(-1)
    permarg = list(range(self.ndim))
    permarg = permarg[1:dim] + [permarg[0]] + permarg[dim+1:] + [permarg[dim]] if dim != 0 else permarg[1:] + [permarg[0]]
    return ((index == Tensor.arange(self.shape[dim], requires_grad=False, device=self.device)) * self.permute(*permarg).shrink(
      tuple([*[(0,sh) for sh in index.shape[1:-1]], None])).unsqueeze(0)).sum(-1, acc_dtype=self.dtype).transpose(0, dim)

  def cat(self:Tensor, *args:Tensor, dim:int=0) -> Tensor:
    """
    Concatenates self with other `Tensor` in `args` along an axis specified by `dim`.
    All tensors must have the same shape except in the concatenating dimension.

    ```python exec="true" source="above" session="tensor" result="python"
    t0, t1, t2 = Tensor([[1, 2]]), Tensor([[3, 4]]), Tensor([[5, 6]])
    print(t0.cat(t1, t2, dim=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t0.cat(t1, t2, dim=1).numpy())
    ```
    """
    dim = self._resolve_dim(dim)
    assert all(len(y.shape) == len(self.shape) and all(y.shape[i] == s for i,s in enumerate(self.shape) if i != dim) for y in args)
    catargs = [self, *args]
    cat_dims = [s.shape[dim] for s in catargs]
    cat_dim_cumsum = [0, *itertools.accumulate(cat_dims)]
    slc:List[List[Optional[Tuple[sint, sint]]]] = [[None for _ in self.shape] for _ in catargs]
    for d,k,s in zip(cat_dims, cat_dim_cumsum[:-1], slc): s[dim] = (k, cat_dim_cumsum[-1] - k - d)
    return functools.reduce(Tensor.__add__, [arg.pad(tuple(s)) for arg,s in zip(catargs, slc)])

  def stack(self:Tensor, *args:Tensor, dim:int=0) -> Tensor:
    """
    Concatenates self with other `Tensor` in `args` along a new dimension specified by `dim`.

    ```python exec="true" source="above" session="tensor" result="python"
    t0, t1, t2 = Tensor([1, 2]), Tensor([3, 4]), Tensor([5, 6])
    print(t0.stack(t1, t2, dim=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t0.stack(t1, t2, dim=1).numpy())
    ```
    """
    # checks for shapes and number of dimensions delegated to cat
    return self.unsqueeze(dim).cat(*[t.unsqueeze(dim) for t in args], dim=dim)

  def repeat(self, repeats, *args) -> Tensor:
    """
    Repeat tensor number of times along each dimension specified by `repeats`.
    `repeats` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3])
    print(t.repeat(4, 2).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.repeat(4, 2, 1).shape)
    ```
    """
    repeats = argfix(repeats, *args)
    base_shape = (1,) * (len(repeats) - self.ndim) + self.shape
    new_shape = [x for b in base_shape for x in [1, b]]
    expand_shape = [x for rs in zip(repeats, base_shape) for x in rs]
    final_shape = [r*s for r,s in zip(repeats, base_shape)]
    return self.reshape(new_shape).expand(expand_shape).reshape(final_shape)

  def _resolve_dim(self, dim:int, *, outer:bool=False) -> int:
    if not -max(1, self.ndim+outer) <= dim < max(1, self.ndim+outer):
      raise IndexError(f"{dim=} out of range {[-max(1, self.ndim+outer), max(1, self.ndim+outer)-1]}")
    return dim + self.ndim+outer if dim < 0 else dim

  def split(self, sizes:Union[int, List[int]], dim:int=0) -> Tuple[Tensor, ...]:
    """
    Splits the tensor into chunks along the dimension specified by `dim`.
    If `sizes` is an integer, it splits into equally sized chunks if possible, otherwise the last chunk will be smaller.
    If `sizes` is a list, it splits into `len(sizes)` chunks with size in `dim` according to `size`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(10).reshape(5, 2)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    split = t.split(2)
    print("\\n".join([repr(x.numpy()) for x in split]))
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    split = t.split([1, 4])
    print("\\n".join([repr(x.numpy()) for x in split]))
    ```
    """
    assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
    dim = self._resolve_dim(dim)
    if isinstance(sizes, int): sizes = [min(sizes, self.shape[dim]-i) for i in range(0, max(1, self.shape[dim]), max(1, sizes))]
    assert sum(sizes) == self.shape[dim], f"expect sizes to sum exactly to {self.shape[dim]}, but got {sum(sizes)}"
    return tuple(self[sl] for sl in [tuple([slice(None)]*dim + [slice(sum(sizes[:i]), sum(sizes[:i + 1]))]) for i in range(len(sizes))])

  def chunk(self, chunks:int, dim:int=0) -> List[Tensor]:
    """
    Splits the tensor into `chunks` number of chunks along the dimension `dim`.
    If the tensor size along `dim` is not divisible by `chunks`, all returned chunks will be the same size except the last one.
    The function may return fewer than the specified number of chunks.

    ```python exec="true" source="above" session="tensor" result="python"
    chunked = Tensor.arange(11).chunk(6)
    print("\\n".join([repr(x.numpy()) for x in chunked]))
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    chunked = Tensor.arange(12).chunk(6)
    print("\\n".join([repr(x.numpy()) for x in chunked]))
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    chunked = Tensor.arange(13).chunk(6)
    print("\\n".join([repr(x.numpy()) for x in chunked]))
    ```
    """
    assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
    assert chunks > 0, f"expect chunks to be greater than 0, got: {chunks}"
    dim = self._resolve_dim(dim)
    return list(self.split(math.ceil(self.shape[dim]/chunks) if self.shape[dim] else [0]*chunks, dim=dim))

  def squeeze(self, dim:Optional[int]=None) -> Tensor:
    """
    Returns a tensor with specified dimensions of input of size 1 removed.
    If `dim` is not specified, all dimensions with size 1 are removed.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.zeros(2, 1, 2, 1, 2)
    print(t.squeeze().shape)
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.squeeze(0).shape)
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.squeeze(1).shape)
    ```
    """
    if dim is None: return self.reshape(tuple(dim for dim in self.shape if dim != 1))
    dim = self._resolve_dim(dim)
    return self if not self.ndim or self.shape[dim] != 1 else self.reshape(self.shape[:dim] + self.shape[dim+1:])

  def unsqueeze(self, dim:int) -> Tensor:
    """
    Returns a tensor with a new dimension of size 1 inserted at the specified `dim`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3, 4])
    print(t.unsqueeze(0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.unsqueeze(1).numpy())
    ```
    """
    dim = self._resolve_dim(dim, outer=True)
    return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])

  def pad2d(self, padding:Sequence[int], value:float=0.0) -> Tensor:
    """
    Returns a tensor that pads the last two axes specified by `padding` (padding_left, padding_right, padding_top, padding_bottom).
    If `value` is specified, the tensor is padded with `value` instead of `0.0`.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(1, 1, 3, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.pad2d((1, 1, 2, 0), value=-float("inf")).numpy())
    ```
    """
    slc = [(-p0, s+p1) for p0,p1,s in zip(padding[::2], padding[1::2], self.shape[::-1])][::-1]
    return self._slice([(0,s) for s in self.shape[:-(len(padding)//2)]] + slc, value=value)

  @property
  def T(self) -> Tensor:
    """`.T` is an alias for `.transpose(1, 0)`."""
    return self.transpose()

  def transpose(self, dim0=1, dim1=0) -> Tensor:
    """
    Returns a tensor that is a transposed version of the original tensor.
    The given dimensions `dim0` and `dim1` are swapped.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.transpose(0, 1).numpy())
    ```
    """
    order = list(range(self.ndim))
    order[dim0], order[dim1] = order[dim1], order[dim0]
    return self.permute(order)

  def flatten(self, start_dim=0, end_dim=-1):
    """
    Flattens the tensor by reshaping it into a one-dimensional tensor.
    If `start_dim` or `end_dim` are passed, only dimensions starting with `start_dim` and ending with `end_dim` are flattened.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(8).reshape(2, 2, 2)
    print(t.flatten().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.flatten(start_dim=1).numpy())
    ```
    """
    start_dim, end_dim = self._resolve_dim(start_dim), self._resolve_dim(end_dim)
    return self.reshape(self.shape[:start_dim] + (prod(self.shape[start_dim:end_dim+1]), ) + self.shape[end_dim+1:])

  def unflatten(self, dim:int, sizes:Tuple[int,...]):
    """
    Expands dimension `dim` of the tensor over multiple dimensions specified by `sizes`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(3, 4, 1).unflatten(1, (2, 2)).shape)
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(3, 4, 1).unflatten(1, (-1, 2)).shape)
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(5, 12, 3).unflatten(-2, (2, 2, 3, 1, 1)).shape)
    ```
    """
    dim = self._resolve_dim(dim)
    return self.reshape(self.shape[:dim] + sizes + self.shape[dim+1:])

  # ***** reduce ops *****

  def _reduce(self, fxn:Type[Function], axis:Optional[Union[int, Tuple[int, ...]]]=None, keepdim=False) -> Tensor:
    if self.ndim == 0:
      if axis is not None and axis not in [-1, 0]: raise IndexError(f"{axis=} out of range of [-1, 0]")
      axis = None
    axis_: Tuple[int, ...] = tuple(range(len(self.shape))) if axis is None else ((axis,) if isinstance(axis, int) else tuple(axis))
    axis_ = tuple(self._resolve_dim(x) for x in axis_)
    ret = fxn.apply(self, axis=axis_)
    return ret if keepdim else ret.reshape(tuple(s for i,s in enumerate(self.shape) if i not in axis_))

  def sum(self, axis=None, keepdim=False, acc_dtype:Optional[DType]=None):
    """
    Sums the elements of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the maximum is computed and whether the reduced dimensions are retained.

    You can pass in `acc_dtype` keyword argument to control the data type of the accumulation.
    If not specified, the accumulation data type is chosen based on the input tensor's data type.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.sum().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.sum(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.sum(axis=1).numpy())
    ```
    """
    ret = self.cast(acc_dtype or sum_acc_dtype(self.dtype))._reduce(F.Sum, axis, keepdim)
    return ret.cast(self.dtype) if self.dtype in {dtypes.float16, dtypes.bfloat16} else ret
  def max(self, axis=None, keepdim=False):
    """
    Returns the maximum value of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the maximum is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 0, 2], [5, 4, 3]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.max().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.max(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.max(axis=1, keepdim=True).numpy())
    ```
    """
    return self._reduce(F.Max, axis, keepdim)
  def min(self, axis=None, keepdim=False):
    """
    Returns the minimum value of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the minimum is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 0, 2], [5, 4, 3]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.min().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.min(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.min(axis=1, keepdim=True).numpy())
    ```
    """
    return -((-self).max(axis=axis, keepdim=keepdim))

  def mean(self, axis=None, keepdim=False):
    """
    Returns the mean value of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the mean is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mean().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mean(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mean(axis=1).numpy())
    ```
    """
    output_dtype = self.dtype if dtypes.is_float(self.dtype) else dtypes.float32
    numerator = self.cast(sum_acc_dtype(self.dtype)).sum(axis=axis, keepdim=keepdim)
    return numerator.div(prod([si for si, so in zip(self.shape, self.sum(axis=axis, keepdim=True).shape) if si != so])).cast(output_dtype)

  def var(self, axis=None, keepdim=False, correction=1):
    """
    Returns the variance of the tensor along the specified axis or axes.

    You can pass in `axis`, `keepdim`, and `correction` keyword arguments to control the axis along
    which the variance is computed, whether the reduced dimensions are retained, and the Bessel's correction applied.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.var().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.var(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.var(axis=1).numpy())
    ```
    """
    assert all_int(self.shape), "does not support symbolic shape"
    square_sum = ((self - self.mean(axis=axis, keepdim=True)).square()).sum(axis=axis, keepdim=keepdim)
    return square_sum.div(max(0, prod(self.shape)/prod(square_sum.shape)-correction))

  def std(self, axis=None, keepdim=False, correction=1):
    """
    Returns the standard deviation of the tensor along the specified axis or axes.

    You can pass in `axis`, `keepdim`, and `correction` keyword arguments to control the axis along
    which the standard deviation is computed, whether the reduced dimensions are retained, and the Bessel's correction applied.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.normal(2, 3, mean=2.5, std=0.5)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.std().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.std(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.std(axis=1).numpy())
    ```
    """
    return self.var(axis, keepdim, correction).sqrt()

  def _softmax(self, axis):
    m = self - self.max(axis=axis, keepdim=True)
    e = m.exp()
    return m, e, e.sum(axis=axis, keepdim=True)

  def softmax(self, axis=-1):
    """
    Applies the softmax function to the tensor along the specified axis.

    Rescales the elements of the tensor such that they lie in the range [0, 1] and sum to 1.

    You can pass in the `axis` keyword argument to control the axis along which the softmax is computed.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.softmax().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.softmax(axis=0).numpy())
    ```
    """
    _, e, ss = self._softmax(axis)
    return e.div(ss)

  def log_softmax(self, axis=-1):
    """
    Applies the log-softmax function to the tensor along the specified axis.

    The log-softmax function is a numerically stable alternative to the softmax function in log space.

    You can pass in the `axis` keyword argument to control the axis along which the log-softmax is computed.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.log_softmax().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.log_softmax(axis=0).numpy())
    ```
    """
    m, _, ss = self._softmax(axis)
    return m - ss.log()

  def logsumexp(self, axis=None, keepdim=False):
    """
    Computes the log-sum-exp of the tensor along the specified axis or axes.

    The log-sum-exp function is a numerically stable way to compute the logarithm of the sum of exponentials.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the log-sum-exp is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logsumexp().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logsumexp(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.logsumexp(axis=1).numpy())
    ```
    """
    m = self.max(axis=axis, keepdim=True)
    return (self - m).exp().sum(axis=axis, keepdim=keepdim).log() + m.squeeze(axis)

  def argmax(self, axis=None, keepdim=False):
    """
    Returns the indices of the maximum value of the tensor along the specified axis.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the maximum is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 0, 2], [5, 4, 3]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmax().numpy()) # Returns the index of the maximum value in the flattened tensor.
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmax(axis=0).numpy()) # Returns the indices of the maximum values along axis 0.
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmax(axis=1).numpy()) # Returns the indices of the maximum values along axis 1.
    ```
    """
    if axis is None:
      idx = (self == self.max(axis)) * Tensor.arange(prod(self.shape)-1,-1,-1, requires_grad=False, device=self.device).reshape(self.shape)
      return (prod(self.shape) - idx.max() - 1).cast(dtypes.int32)
    axis = self._resolve_dim(axis)
    m = self == self.max(axis=axis, keepdim=True)
    idx = m * Tensor.arange(self.shape[axis]-1,-1,-1, requires_grad=False, device=self.device).reshape(self.shape[axis], *[1]*(self.ndim-axis-1))
    return (self.shape[axis]-idx.max(axis=axis, keepdim=keepdim)-1).cast(dtypes.int32)

  def argmin(self, axis=None, keepdim=False):
    """
    Returns the indices of the minimum value of the tensor along the specified axis.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the minimum is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 0, 2], [5, 4, 3]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmin().numpy()) # Returns the index of the minimum value in the flattened tensor.
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmin(axis=0).numpy()) # Returns the indices of the minimum values along axis 0.
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.argmin(axis=1).numpy()) # Returns the indices of the minimum values along axis 1.
    ```
    """
    return (-self).argmax(axis=axis, keepdim=keepdim)

  @staticmethod
  def einsum(formula:str, *raw_xs, acc_dtype:Optional[DType]=None) -> Tensor:
    """
    Sums the product of the elements of the input tensors according to a formula based on the Einstein summation convention.

    See: https://pytorch.org/docs/stable/generated/torch.einsum.html

    ```python exec="true" source="above" session="tensor" result="python"
    x = Tensor([[1, 2], [3, 4]])
    y = Tensor([[5, 6], [7, 8]])
    print(Tensor.einsum("ij,ij->", x, y).numpy())
    ```
    """
    xs:Tuple[Tensor] = argfix(*raw_xs)
    formula = formula.replace(" ", "")
    inputs_str, output = formula.split("->") if "->" in formula else (formula, sorted(formula))
    inputs = [x for x in cast(str,inputs_str).split(',')]
    assert len(xs) == len(inputs), f"number of inputs doesn't match number of operands in formula, expected {len(inputs)}, got {len(xs)}"

    # map the value of each letter in the formula
    letter_val = sorted(merge_dicts([{letter:dim for letter, dim in zip(letters, tensor.shape)} for letters, tensor in zip(inputs, xs)]).items())

    xs_:List[Tensor] = []
    lhs = [sorted(enumerate(s), key=lambda e:e[1]) for s in inputs]
    for x,(order,letters) in zip(xs, [list(zip(*l)) for l in lhs]):
      # permute to the sorted letter order, then reshape/expand to create dimensions for the missing letters
      xs_.append(x.permute(order).reshape([val if letter in letters else 1 for letter,val in letter_val]).expand([val for _,val in letter_val]))

    # determine the inverse permutation to revert back to original order
    rhs_letter_order = argsort(list(output))
    rhs_order = argsort(rhs_letter_order)

    # sum over all axes that's not in the output, then permute to the output order
    return functools.reduce(lambda a,b:a*b, xs_) \
      .sum(axis=[axis for axis,(letter,_) in enumerate(letter_val) if letter not in output],acc_dtype=acc_dtype).permute(rhs_order)

  # ***** processing ops *****

  def _pool(self, k_:Tuple[sint, ...], stride:Union[Tuple[int, ...], int]=1, dilation:Union[Tuple[int, ...], int]=1) -> Tensor:
    assert len(self.shape) >= len(k_), f"can't pool {self.shape} with {k_}"
    assert all_int(self.shape) and all_int(k_), f"does not support symbolic {self.shape=}, {k_=}"
    s_, d_ = make_pair(stride, len(k_)), make_pair(dilation, len(k_))
    assert len(k_) == len(s_) == len(d_), f"stride/dilation mismatch kernel:{k_} stride:{s_} dilation:{d_}"
    noop_, i_ = [None] * len(self.shape[:-len(k_)]), self.shape[-len(k_):]
    if any(k > s for k,s in zip(k_, s_)) or any(d != 1 for d in d_):
      o_ = [(i - d * (k-1) - 1)//s + 1 for i,d,k,s in zip(i_, d_, k_, s_)]
      # repeats such that we don't need padding
      xup = self.repeat([1]*len(noop_) + [math.ceil(k*(i+d) / i) for k,i,d in zip(k_, i_, d_)])
      # slice by dilation
      xup = xup.shrink(tuple(noop_ + [(0,k*(i+d)) for k,i,d in zip(k_, i_, d_)])).reshape(noop_ + flatten((k,i+d) for k,i,d in zip(k_, i_, d_)))
      # handle stride
      xup = xup.shrink(noop_ + flatten(((0,k), (0,o*s)) for k,o,s in zip(k_, o_, s_))).reshape(noop_ + flatten((k,o,s) for k,o,s in zip(k_, o_, s_)))
      xup = xup.shrink(noop_ + flatten(((0,k), (0,o), (0,1)) for k,o in zip(k_, o_))).reshape(noop_ + flatten((k,o) for k,o in zip(k_, o_)))
      # permute to move reduce to the end
      return xup.permute(*range(len(noop_)), *[len(noop_)+i*2+1 for i in range(len(i_))], *[len(noop_)+i*2 for i in range(len(i_))])
    # TODO: once the shapetracker can optimize well, remove this alternative implementation. or not if the CPU implementation doesn't use ShapeTracker
    o_ = [(i+(s-k))//s for i,s,k in zip(i_, s_, k_)]
    xup = self.pad(tuple(noop_ + [(0, max(0,o*s-i)) for i,o,s in zip(i_, o_, s_)])).shrink(tuple(noop_ + [(0,o*s) for o,s in zip(o_, s_)]))
    xup = xup.reshape(noop_ + flatten(((o,s) for o,s in zip(o_, s_))))
    xup = xup.shrink(noop_ + flatten(((0,o), (0,k)) for o,k in zip(o_, k_)))
    return xup.permute(*range(len(noop_)), *[len(noop_)+i*2 for i in range(len(i_))], *[len(noop_)+i*2+1 for i in range(len(i_))])

  # NOTE: these work for more than 2D
  def avg_pool2d(self, kernel_size=(2,2), stride=None, dilation=1):
    """
    Applies average pooling over a tensor.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d pooling and instead works for any number of dimensions.

    See: https://paperswithcode.com/method/average-pooling

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(1, 1, 3, 3)
    print(t.avg_pool2d().numpy())
    ```
    """
    return self._pool(
      make_pair(kernel_size), stride if stride is not None else kernel_size, dilation).mean(axis=tuple(range(0-len(make_pair(kernel_size)), 0)))
  def max_pool2d(self, kernel_size=(2,2), stride=None, dilation=1):
    """
    Applies max pooling over a tensor.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d pooling and instead works for any number of dimensions.

    See: https://paperswithcode.com/method/max-pooling

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(1, 1, 3, 3)
    print(t.max_pool2d().numpy())
    ```
    """
    return self._pool(
      make_pair(kernel_size), stride if stride is not None else kernel_size, dilation).max(axis=tuple(range(0-len(make_pair(kernel_size)), 0)))

  def conv2d(self, weight:Tensor, bias:Optional[Tensor]=None, groups=1, stride=1, dilation=1, padding=0, acc_dtype:Optional[DType]=None) -> Tensor:
    """
    Applies a convolution over a tensor with a given weight and optional bias.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d convolutions and instead works for any number of dimensions.

    See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(1, 1, 3, 3)
    w = Tensor.ones(1, 1, 2, 2)
    print(t.conv2d(w).numpy())
    ```
    """
    (bs,cin_), (cout,cin), HW = self.shape[:2], weight.shape[:2], weight.shape[2:]
    assert groups*cin == cin_ and len(self.shape) == len(weight.shape), f"Input Tensor shape {self.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"  # noqa: E501
    if isinstance(padding, (tuple,list)): assert len(padding) == 2*len(HW) or len(padding) == len(HW), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {self.shape}"  # noqa: E501
    padding_ = [padding]*2*len(HW) if isinstance(padding, int) else (padding if len(padding) == 2*len(HW) else [p for p in padding for _ in range(2)][::-1])  # noqa: E501

    # conv2d is a pooling op (with padding)
    x = self.pad2d(padding_)._pool(HW, stride, dilation)   # (bs, groups*cin, oy, ox, H, W)
    rcout, oyx = cout//groups, x.shape[2:-len(HW)]
    if not all(x == 3 for x in HW) or stride != 1 or dilation != 1 or not WINO:
      # normal conv
      x = x.reshape(bs, groups, cin, 1, *oyx, *HW).expand(bs, groups, cin, rcout, *oyx, *HW).permute(0,1,3,*[4+i for i in range(len(oyx))],2,*[4+len(oyx)+i for i in range(len(HW))])  # noqa: E501

      # conv! broadcasted to (bs, groups, rcout, *oyx, cin, *HW)
      ret = (x * weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW)).sum([-1-i for i in range(1+len(oyx))], keepdim=True, acc_dtype=acc_dtype).reshape(bs, cout, *oyx)  # noqa: E501
      return ret if bias is None else ret.add(bias.reshape(1, -1, *[1] * len(HW)))

    HWI, HWO = (6,) * len(HW), (4,) * len(HW)  # F(4x4,3x3) winograd tiles
    winograd_G = [[1/4, 0, 0], [-1/6, -1/6, -1/6], [-1/6, 1/6, -1/6], [1/24, 1/12, 1/6], [1/24, -1/12, 1/6], [0, 0, 1]]
    winograd_Bt = [[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0], [0, -2, -1, 2, 1, 0], [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]]
    winograd_At = [[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0], [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]] # applying At in pre-order doubles compile time

    # todo: stride == dilation
    # use padding to round up to 4x4 output tiles
    # (bs, cin_, tyx, HWI)
    d = self.pad2d(sum([[padding_[i*2], padding_[i*2+1] + (-(dim + sum(padding_[i * 2:(i + 1) * 2]) - 2) % 4)] for i, dim in enumerate(self.shape[-len(HW):])], []))._pool(HWI, HWO)  # noqa: E501
    # move HW to the front: # (HWI, bs, cin_, tyx)
    d = d.permute(*range(len(d.shape)-len(HW),len(d.shape)), *range(len(d.shape)-len(HW)))
    tyx = d.shape[-len(HWI):]  # dim of tiling

    g = weight.permute(*range(len(weight.shape)-len(HW),len(weight.shape)), *range(len(weight.shape)-len(HW)))  # move HW to the front

    # compute 6x6 winograd tiles: GgGt, BtdB
    # (HWI, groups * rcout, cin) -> (HWI, bs=1, groups, rcout, cin, tyx=(1,1))
    gfactors = _apply_winograd_matrix(winograd_G, g, len(HW)).reshape(*HWI, 1, groups, rcout, cin, *([1]*len(tyx)))
    # (HWI, bs, cin_, tyx) -> (HWI, bs, groups, 1 ,cin, *tyx)
    dfactors = _apply_winograd_matrix(winograd_Bt, d, len(HW)).reshape(*HWI, bs, groups, 1, cin, *tyx)

    # matmul; sum across cin: (HWI, bs, groups, rcout, *tyx); then HWI -> HWO: (HWO, bs, groups, rcout, *tyx)
    ret = _apply_winograd_matrix(winograd_At, (gfactors * dfactors).sum(axis=-1-len(HW), acc_dtype=acc_dtype), len(HW))

    # interleave tyx and HWO: (bs, groups, rcout, oy, HO, ox, WO)
    ret = ret.permute([*range(len(HW), len(ret.shape)-len(HW)), *[i+o for i in range(len(HW)) for o in [len(ret.shape)-len(HW),0]]])
    # merge groups and rcout, tyx and HWO: (bs, groups, cout, *yx), shrink to final
    ret = ret.reshape(bs, cout, *[c * HWO[i] for i, c in enumerate(tyx)]).shrink(tuple((0, s) for s in [bs, cout, *oyx]))

    return (ret if bias is None else ret.add(bias.reshape(1, -1, *[1 for _ in range(len(HW))]))).contiguous().contiguous_backward()

  def conv_transpose2d(self, weight:Tensor, bias:Optional[Tensor]=None, groups=1, stride=1, dilation=1, padding=0, output_padding=0) -> Tensor:
    """
    Applies a transposed convolution over a tensor with a given weight and optional bias.

    NOTE: unlike PyTorch, this implementation is not limited to only 2d transposed convolutions and instead works for any number of dimensions.

    See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(9).reshape(1, 1, 3, 3)
    w = Tensor.ones(1, 1, 2, 2)
    print(t.conv_transpose2d(w).numpy())
    ```
    """
    HW, trailing = weight.shape[2:], list(range(3, len(weight.shape)+1))
    x, w = self, weight.unflatten(0, (groups, -1)).permute(0,2,1,*trailing).flip(trailing)
    stride = make_pair(stride, len(HW))
    if any(s>1 for s in stride):
      x = x.reshape(None, None, *flatten((k,1) for k in x.shape[2:]))
      x = x.pad((None, None, *flatten((None,(0,s-1)) for s in stride)))
      x = x.reshape(None, None, *[k*s for k,s in zip(x.shape[2::2], stride)])
      x = x.shrink((None, None, *[(0,k-(s-1)) for k,s in zip(x.shape[2:], stride)]))
    padding = flatten((((k-1)*d-p,(k-1)*d-p+op) for k,d,p,op in reversed(list(
      zip(HW, make_pair(dilation, len(HW)), make_pair(padding, len(HW)), make_pair(output_padding, len(HW)))))))
    return x.conv2d(w.flatten(end_dim=1), groups=groups, bias=bias, dilation=dilation, padding=padding)

  def dot(self, w:Tensor, acc_dtype:Optional[DType]=None) -> Tensor:
    """
    Performs dot product between two tensors.

    ```python exec="true" source="above" session="tensor" result="python"
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    print(a.dot(b).numpy())
    ```
    """
    n1, n2 = len(self.shape), len(w.shape)
    assert n1 != 0 and n2 != 0, f"both arguments to matmul need to be at least 1D, but they are {n1}D and {n2}D"
    assert (L:=self.shape[-1]) == (R:=w.shape[-min(n2, 2)]), f"Input Tensor shapes {self.shape} and {w.shape} cannot be multiplied ({L} != {R})"
    x = self.reshape(*self.shape[0:-1], *[1]*min(n1-1, n2-1, 1), self.shape[-1])
    w = w.reshape(*w.shape[0:-2], *[1]*min(n1-1, n2-1, 1), *w.shape[-min(n2, 2):]).transpose(-1, -min(n2, 2))
    return (x*w).sum(-1, acc_dtype=acc_dtype).cast(least_upper_dtype(x.dtype, w.dtype))

  def matmul(self, x:Tensor, reverse=False, acc_dtype:Optional[DType]=None) -> Tensor:
    """
    Performs matrix multiplication between two tensors.

    You can pass in the `reverse` keyword argument to control the order of the matrix multiplication.
    You can pass in the optional `acc_dtype` keyword argument to control the data type of the accumulation.

    ```python exec="true" source="above" session="tensor" result="python"
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    print(a.matmul(b).numpy())
    ```
    """
    return x.dot(self, acc_dtype=acc_dtype) if reverse else self.dot(x, acc_dtype=acc_dtype)

  def _cumsum(self, axis:int=0, _first_zero=False) -> Tensor:
    pl_sz = self.shape[axis] - int(not _first_zero and self.shape[axis] != 0)
    return self.transpose(axis,-1).pad2d((pl_sz,0))._pool((self.shape[axis] or 1,)).sum(-1).transpose(axis,-1)
  def cumsum(self, axis:int=0) -> Tensor:
    """
    Computes the cumulative sum of the tensor along the specified axis.

    You can pass in the `axis` keyword argument to control the axis along which the cumulative sum is computed.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.cumsum(1).numpy())
    ```
    """
    # TODO: someday the optimizer will find this on it's own
    # for now this is a two stage cumsum
    SPLIT = 256
    if self.shape[axis] <= SPLIT*2: return self._cumsum(axis)
    ret = self.transpose(axis,-1).pad2d((round_up(self.shape[axis], SPLIT)-self.shape[axis], 0))
    ret = ret.unflatten(-1, (-1, SPLIT))._cumsum(-1)
    base_add = ret[..., -1]._cumsum(-1, _first_zero=True)[..., :-1]
    base_add = base_add.unsqueeze(-1).expand(*base_add.shape, ret.shape[-1])
    def fix(x:Tensor): return x.flatten(start_dim=-2)[..., -self.shape[axis]:].transpose(axis,-1)
    return fix(ret) + fix(base_add)

  @staticmethod
  def _tri(r:sint, c:sint, k:int=0, **kwargs) -> Tensor:
    assert all_int((r,c)), "does not support symbolic"
    if r == 0: return Tensor.zeros((r, c), **kwargs)
    return Tensor.arange(r, **kwargs).unsqueeze(1).expand(r,c) <= Tensor.arange(-k, c-k, **kwargs).unsqueeze(0).expand(r,c)
  def triu(self, k:int=0) -> Tensor:
    """
    Returns the upper triangular part of the tensor, the other elements are set to 0.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.triu(k=1).numpy())
    ```
    """
    return Tensor._tri(self.shape[-2], self.shape[-1], k=k, device=self.device).where(self, 0)
  def tril(self, k:int=0) -> Tensor:
    """
    Returns the lower triangular part of the tensor, the other elements are set to 0.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.tril().numpy())
    ```
    """
    return Tensor._tri(self.shape[-2], self.shape[-1], k=k+1, device=self.device).where(0, self)

  # ***** unary ops *****

  def logical_not(self):
    """
    Computes the logical NOT of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([False, True]).logical_not().numpy())
    ```
    """
    return F.Eq.apply(*self._broadcasted(False))
  def neg(self):
    """
    Negates the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).neg().numpy())
    ```
    """
    return F.Neg.apply(self) if self.dtype != dtypes.bool else self.logical_not()
  def contiguous(self):
    """
    Returns a contiguous tensor.
    """
    return F.Contiguous.apply(self)
  def contiguous_backward(self):
    """
    Inserts a contiguous operation in the backward pass.
    """
    return F.ContiguousBackward.apply(self)
  def log(self):
    """
    Computes the natural logarithm element-wise.

    See: https://en.wikipedia.org/wiki/Logarithm

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 4., 8.]).log().numpy())
    ```
    """
    return F.Log.apply(self.cast(least_upper_float(self.dtype)))
  def log2(self):
    """
    Computes the base-2 logarithm element-wise.

    See: https://en.wikipedia.org/wiki/Logarithm

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 4., 8.]).log2().numpy())
    ```
    """
    return self.log()/math.log(2)
  def exp(self):
    """
    Computes the exponential function element-wise.

    See: https://en.wikipedia.org/wiki/Exponential_function

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., 1., 2., 3.]).exp().numpy())
    ```
    """
    return F.Exp.apply(self.cast(least_upper_float(self.dtype)))
  def exp2(self):
    """
    Computes the base-2 exponential function element-wise.

    See: https://en.wikipedia.org/wiki/Exponential_function

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., 1., 2., 3.]).exp2().numpy())
    ```
    """
    return F.Exp.apply(self*math.log(2))
  def relu(self):
    """
    Applies the Rectified Linear Unit (ReLU) function element-wise.

    - Described: https://paperswithcode.com/method/relu

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).relu().numpy())
    ```
    """
    return F.Relu.apply(self)
  def sigmoid(self):
    """
    Applies the Sigmoid function element-wise.

    - Described: https://en.wikipedia.org/wiki/Sigmoid_function

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sigmoid().numpy())
    ```
    """
    return F.Sigmoid.apply(self.cast(least_upper_float(self.dtype)))
  def sqrt(self):
    """
    Computes the square root of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 3.]).sqrt().numpy())
    ```
    """
    return F.Sqrt.apply(self.cast(least_upper_float(self.dtype)))
  def rsqrt(self):
    """
    Computes the reciprocal of the square root of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 3.]).rsqrt().numpy())
    ```
    """
    return self.reciprocal().sqrt()
  def sin(self):
    """
    Computes the sine of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]).sin().numpy())
    ```
    """
    return F.Sin.apply(self.cast(least_upper_float(self.dtype)))
  def cos(self):
    """
    Computes the cosine of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]).cos().numpy())
    ```
    """
    return ((math.pi/2)-self).sin()
  def tan(self):
    """
    Computes the tangent of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([0., math.pi/4, math.pi/2, 3*math.pi/4, math.pi]).tan().numpy())
    ```
    """
    return self.sin() / self.cos()

  # ***** math functions *****

  def trunc(self: Tensor) -> Tensor:
    """
    Truncates the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3.9, -2.1, -1.5, 0.5, 1.5, 2.1, 3.9]).trunc().numpy())
    ```
    """
    return self.cast(dtypes.int32).cast(self.dtype)
  def ceil(self: Tensor) -> Tensor:
    """
    Rounds the tensor element-wise towards positive infinity.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3.9, -2.1, -1.5, 0.5, 1.5, 2.1, 3.9]).ceil().numpy())
    ```
    """
    return (self > (b := self.trunc())).where(b+1, b)
  def floor(self: Tensor) -> Tensor:
    """
    Rounds the tensor element-wise towards negative infinity.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3.9, -2.1, -1.5, 0.5, 1.5, 2.1, 3.9]).floor().numpy())
    ```
    """
    return (self < (b := self.trunc())).where(b-1, b)
  def round(self: Tensor) -> Tensor:
    """
    Rounds the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3.9, -2.1, -1.5, 0.5, 1.5, 2.1, 3.9]).round().numpy())
    ```
    """
    return ((self > 0) == ((b := self.cast(dtypes.int32) / 2.0).cast(dtypes.int32) == b)).where((self - 0.5).ceil(), (self + 0.5).floor())
  def lerp(self, end: Tensor, weight: Union[Tensor, float]) -> Tensor:
    """
    Linearly interpolates between `self` and `end` by `weight`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 3.]).lerp(Tensor([4., 5., 6.]), 0.5).numpy())
    ```
    """
    return self + (end - self) * weight
  def square(self):
    """
    Convenience method for squaring the tensor element-wise.
    Equivalent to `self*self`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).square().numpy())
    ```
    """
    return self*self
  def clip(self, min_, max_):
    """
    Clips (limits) the values in the tensor between `min_` and `max_` element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).clip(-1, 1).numpy())
    ```
    """
    return self.maximum(min_).minimum(max_)
  def sign(self):
    """
    Returns the sign of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sign().numpy())
    ```
    """
    return F.Sign.apply(self)
  def abs(self):
    """
    Computes the absolute value of the tensor element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).abs().numpy())
    ```
    """
    return self * self.sign()
  def reciprocal(self):
    """
    Compute `1/x` element-wise.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1., 2., 3.]).reciprocal().numpy())
    ```
    """
    return F.Reciprocal.apply(self.cast(least_upper_float(self.dtype)))

  # ***** activation functions *****

  def elu(self, alpha=1.0):
    """
    Applies the Exponential Linear Unit (ELU) function element-wise.

    - Described: https://paperswithcode.com/method/elu
    - Paper: https://arxiv.org/abs/1511.07289v5

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).elu().numpy())
    ```
    """
    return self.relu() - alpha*(1-self.exp()).relu()

  def celu(self, alpha=1.0):
    """
    Applies the Continuously differentiable Exponential Linear Unit (CELU) function element-wise.

    - Described: https://paperswithcode.com/method/celu
    - Paper: https://arxiv.org/abs/1704.07483

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).celu().numpy())
    ```
    """
    return self.maximum(0) + (alpha * ((self / alpha).exp() - 1)).minimum(0)

  def swish(self):
    """
    See `.silu()`

    - Paper: https://arxiv.org/abs/1710.05941v1

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).swish().numpy())
    ```
    """
    return self * self.sigmoid()

  def silu(self):
    """
    Applies the Sigmoid Linear Unit (SiLU) function element-wise.

    - Described: https://paperswithcode.com/method/silu
    - Paper: https://arxiv.org/abs/1606.08415

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).silu().numpy())
    ```
    """
    return self.swish()   # The SiLU function is also known as the swish function.

  def relu6(self):
    """
    Applies the ReLU6 function element-wise.

    - Described: https://paperswithcode.com/method/relu6
    - Paper: https://arxiv.org/abs/1704.04861v1

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-9., -6., -3., 0., 3., 6., 9.]).relu6().numpy())
    ```
    """
    return self.relu() - (self-6).relu()

  def hardswish(self):
    """
    Applies the Hardswish function element-wise.

    - Described: https://paperswithcode.com/method/hard-swish
    - Paper: https://arxiv.org/abs/1905.02244v5

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).hardswish().numpy())
    ```
    """
    return self * (self+3).relu6() * (1/6)

  def tanh(self):
    """
    Applies the Hyperbolic Tangent (tanh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Tanh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).tanh().numpy())
    ```
    """
    return 2.0 * ((2.0 * self).sigmoid()) - 1.0

  def sinh(self):
    """
    Applies the Hyperbolic Sine (sinh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Sinh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).sinh().numpy())
    ```
    """
    return (self.exp() - self.neg().exp()) / 2

  def cosh(self):
    """
    Applies the Hyperbolic Cosine (cosh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions#Cosh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).cosh().numpy())
    ```
    """
    return (self.exp() + self.neg().exp()) / 2

  def atanh(self):
    """
    Applies the Inverse Hyperbolic Tangent (atanh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#atanh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9]).atanh().numpy())
    ```
    """
    return ((1 + self)/(1 - self)).log() / 2

  def asinh(self):
    """
    Applies the Inverse Hyperbolic Sine (asinh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#asinh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).asinh().numpy())
    ```
    """
    return (self + (self.square() + 1).sqrt()).log()

  def acosh(self):
    """
    Applies the Inverse Hyperbolic Cosine (acosh) function element-wise.

    - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#acosh

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).acosh().numpy())
    ```
    """
    return (self + (self.square() - 1).sqrt()).log()

  def hardtanh(self, min_val=-1, max_val=1):
    """
    Applies the Hardtanh function element-wise.

    - Described: https://paperswithcode.com/method/hardtanh-activation

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5]).hardtanh().numpy())
    ```
    """
    return self.clip(min_val, max_val)

  def gelu(self):
    """
    Applies the Gaussian Error Linear Unit (GELU) function element-wise.

    - Described: https://paperswithcode.com/method/gelu
    - Paper: https://arxiv.org/abs/1606.08415v5

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).gelu().numpy())
    ```
    """
    return 0.5 * self * (1 + (self * 0.7978845608 * (1 + 0.044715 * self * self)).tanh())

  def quick_gelu(self):
    """
    Applies the Sigmoid GELU approximation element-wise.

    - Described: https://paperswithcode.com/method/gelu

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).quick_gelu().numpy())
    ```
    """
    return self * (self * 1.702).sigmoid()

  def leakyrelu(self, neg_slope=0.01):
    """
    Applies the Leaky ReLU function element-wise.

    - Described: https://paperswithcode.com/method/leaky-relu

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).leakyrelu().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).leakyrelu(neg_slope=0.42).numpy())
    ```
    """
    return self.relu() - (-neg_slope*self).relu()

  def mish(self):
    """
    Applies the Mish function element-wise.

    - Described: https://paperswithcode.com/method/mish
    - Paper: https://arxiv.org/abs/1908.08681v3

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).mish().numpy())
    ```
    """
    return self * self.softplus().tanh()

  def softplus(self, beta=1):
    """
    Applies the Softplus function element-wise.

    - Described: https://paperswithcode.com/method/softplus

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).softplus().numpy())
    ```
    """
    return (1/beta) * (1 + (self*beta).exp()).log()

  def softsign(self):
    """
    Applies the Softsign function element-wise.

    - Described: https://paperswithcode.com/method/softsign

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-3., -2., -1., 0., 1., 2., 3.]).softsign().numpy())
    ```
    """
    return self / (1 + self.abs())

  # ***** broadcasted elementwise ops *****
  def _broadcast_to(self, shape:Tuple[sint, ...]):
    reshape_arg, _ = _pad_left(self.shape, shape)
    if self.ndim > len(shape) or not all(sh in {s,1} or (s==0 and sh==1) for sh,s in zip(reshape_arg, shape)):
      raise ValueError(f"cannot broadcast tensor with shape={self.shape} to {shape=}")
    return F.Expand.apply(self.reshape(reshape_arg), shape=shape) if shape != self.shape else self

  def _broadcasted(self, y:Union[Tensor, ConstType], reverse:bool=False, match_dtype:bool=True) -> Tuple[Tensor, Tensor]:
    x: Tensor = self
    if not isinstance(y, Tensor):
      # make y a Tensor
      assert isinstance(y, (float, int, bool, Node)), f"{type(y)=}, {y=}"
      if isinstance(self.dtype, ImageDType) or dtypes.is_float(x.dtype) or (dtypes.is_int(x.dtype) and isinstance(y, int)): y_dtype = x.dtype
      else: y_dtype = dtypes.from_py(y)
      if isinstance(y, Node): y = Tensor.from_node(y, device=self.device)
      else: y = Tensor(dtypes.as_const(y, y_dtype), self.device, y_dtype, requires_grad=False)

    if match_dtype:
      output_dtype = least_upper_dtype(x.dtype, y.dtype)
      x, y = x.cast(output_dtype), y.cast(output_dtype)

    if reverse: x, y = y, x

    # broadcast
    out_shape = _broadcast_shape(x.shape, y.shape)
    return x._broadcast_to(out_shape), y._broadcast_to(out_shape)

  def _to_const_val(self, x:Union[Tensor, ConstType]) -> Union[Tensor, ConstType]:
    # TODO: update with multi
    return x.lazydata.base.arg if isinstance(x, Tensor) and isinstance(x.lazydata, LazyBuffer) and x.lazydata.is_unrealized_unmasked_const() \
      and not x.requires_grad and self._broadcasted(x)[0].shape == self.shape else x

  def add(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Adds `self` and `x`.
    Equivalent to `self + x`.
    Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(4)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.add(20).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.add(Tensor([[2.0], [3.5]])).numpy())
    ```
    """
    return F.Add.apply(*self._broadcasted(x, reverse))

  def sub(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Subtracts `x` from `self`.
    Equivalent to `self - x`.
    Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(4)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.sub(20).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.sub(Tensor([[2.0], [3.5]])).numpy())
    ```
    """
    return F.Sub.apply(*self._broadcasted(x, reverse))

  def mul(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Multiplies `self` and `x`.
    Equivalent to `self * x`.
    Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(4)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mul(3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.mul(Tensor([[-1.0], [2.0]])).numpy())
    ```
    """
    return F.Mul.apply(*self._broadcasted(x, reverse))

  def div(self, x:Union[Tensor, ConstType], reverse=False, upcast=True) -> Tensor:
    """
    Divides `self` by `x`.
    Equivalent to `self / x`.
    Supports broadcasting to a common shape, type promotion, and integer, float, boolean inputs.
    By default, `div` performs true division. Set `upcast` to `False` for integer division.

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(4)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.div(3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, 4, 10]).div(Tensor([2, 3, 4])).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, 4, 10]).div(Tensor([2, 3, 4]), upcast=False).numpy())
    ```
    """
    numerator, denominator = self._broadcasted(x, reverse)
    if upcast: numerator, denominator = numerator.cast(least_upper_float(numerator.dtype)), denominator.cast(least_upper_float(denominator.dtype))
    return F.Div.apply(numerator, denominator)

  def xor(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Computes bitwise xor of `self` and `x`.
    Equivalent to `self ^ x`.
    Supports broadcasting to a common shape, type promotion, and integer, boolean inputs.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, -2, 3]).xor(Tensor([1, 0, 3])).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([True, True, False, False]).xor(Tensor([True, False, True, False])).numpy())
    ```
    """
    return F.Xor.apply(*self._broadcasted(x, reverse))

  def lshift(self, x:int):
    """
    Computes left arithmetic shift of `self` by `x` bits. `self` must have unsigned dtype.
    Equivalent to `self << x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([1, 3, 31], dtype=dtypes.uint8).lshift(2).numpy())
    ```
    """
    assert dtypes.is_unsigned(self.dtype) and isinstance(x, int) and x >= 0, f"not supported {self.dtype=} {x=}"
    return self.mul(2 ** x)

  def rshift(self, x:int):
    """
    Computes right arithmetic shift of `self` by `x` bits. `self` must have unsigned dtype.
    Equivalent to `self >> x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([4, 13, 125], dtype=dtypes.uint8).rshift(2).numpy())
    ```
    """
    assert dtypes.is_unsigned(self.dtype) and isinstance(x, int) and x >= 0, f"not supported {self.dtype=} {x=}"
    return self.div(2 ** x, upcast=False)

  def pow(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
    """
    Computes power of `self` with `x`.
    Equivalent to `self ** x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).pow(2).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).pow(Tensor([-1.5, 0.5, 1.5])).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print((2 ** Tensor([-1, 2, 3])).numpy())
    ```
    """
    x = self._to_const_val(x)
    if not isinstance(x, Tensor) and not reverse:
      # simple pow identities
      if x < 0: return self.reciprocal().pow(-x)
      if x == 0: return 1 + self * 0
      if int(x - 0.5) + 0.5 == x: return self.pow(int(x - 0.5)) * self.sqrt()
      if int(x) == x: return self.pow(x // 2).square() * (1 if x % 2 == 0 else self)

    # positive const ** self
    if not isinstance(x, Tensor) and reverse and x > 0: return self.mul(math.log(x)).exp()

    base, exponent = self._broadcasted(x, reverse=reverse)
    # start with b ** e = exp(e * log(b))
    ret = base.abs().log().mul(exponent).exp()
    # correct sign of negative base with odd exponent (cos has a period of 2pi so we use it here to get the oddness of the exponent)
    negative_base = (base < 0).detach().where(1, 0)
    # 1 for non-negative base or negative even exponent, -1 for negative odd exponent, don't care about non-integer exponent
    correct_sign = 1 + negative_base * ((exponent * math.pi).cos() - 1)
    # inject nan for negative base and non-integer exponent
    inject_nan = (negative_base * (exponent != exponent.trunc())).detach().where(math.nan, 1)
    # apply correct_sign inject_nan, and fix 0 ** 0 = 1
    return ((base == 0) * (exponent == 0)).detach().where(1, ret * correct_sign * inject_nan)

  def maximum(self, x:Union[Tensor, ConstType]) -> Tensor:
    """
    Computes element-wise maximum of `self` and `x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).maximum(1).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).maximum(Tensor([-4, -2, 9])).numpy())
    ```
    """
    return (self<x).detach().where(x, (self==x).detach().where(((self * 0.5 + x * 0.5).cast(self.dtype)), self))

  def minimum(self, x:Union[Tensor, ConstType]) -> Tensor:
    """
    Computes element-wise minimum of `self` and `x`.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).minimum(1).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor([-1, 2, 3]).minimum(Tensor([-4, -2, 9])).numpy())
    ```
    """
    return -((-self).maximum(-x))

  def where(self:Tensor, x:Union[Tensor, ConstType], y:Union[Tensor, ConstType]):
    """
    Return a tensor of elements selected from either `x` or `y`, depending on `self`.
    `output_i = x_i if self_i else y_i`.

    ```python exec="true" source="above" session="tensor" result="python"
    cond = Tensor([[True, True, False], [True, False, False]])
    print(cond.where(1, 3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    cond = Tensor.randn(2, 3)
    print(cond.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print((cond > 0).where(cond, -float("inf")).numpy())
    ```
    """
    if isinstance(x, Tensor): x, y = x._broadcasted(y)
    elif isinstance(y, Tensor): y, x = y._broadcasted(x)
    cond, x = self._broadcasted(x, match_dtype=False)
    cond, y = cond._broadcasted(y, match_dtype=False)
    return F.Where.apply(cond.cast(dtypes.bool), *x._broadcasted(y))

  def masked_fill(self:Tensor, mask:Tensor, value:Union[Tensor, ConstType]): return mask.where(value, self)

  # ***** op wrappers *****

  def __neg__(self) -> Tensor: return self.neg()

  def __add__(self, x) -> Tensor: return self.add(x)
  def __sub__(self, x) -> Tensor: return self.sub(x)
  def __mul__(self, x) -> Tensor: return self.mul(x)
  def __pow__(self, x) -> Tensor: return self.pow(x)
  def __truediv__(self, x) -> Tensor: return self.div(x)
  def __matmul__(self, x) -> Tensor: return self.matmul(x)
  def __xor__(self, x) -> Tensor: return self.xor(x)
  def __lshift__(self, x) -> Tensor: return self.lshift(x)
  def __rshift__(self, x) -> Tensor: return self.rshift(x)

  def __radd__(self, x) -> Tensor: return self.add(x, True)
  def __rsub__(self, x) -> Tensor: return self.sub(x, True)
  def __rmul__(self, x) -> Tensor: return self.mul(x, True)
  def __rpow__(self, x) -> Tensor: return self.pow(x, True)
  def __rtruediv__(self, x) -> Tensor: return self.div(x, True)
  def __rmatmul__(self, x) -> Tensor: return self.matmul(x, True)
  def __rxor__(self, x) -> Tensor: return self.xor(x, True)

  def __iadd__(self, x) -> Tensor: return self.assign(self.add(x))
  def __isub__(self, x) -> Tensor: return self.assign(self.sub(x))
  def __imul__(self, x) -> Tensor: return self.assign(self.mul(x))
  def __ipow__(self, x) -> Tensor: return self.assign(self.pow(x))
  def __itruediv__(self, x) -> Tensor: return self.assign(self.div(x))
  def __imatmul__(self, x) -> Tensor: return self.assign(self.matmul(x))
  def __ixor__(self, x) -> Tensor: return self.assign(self.xor(x))
  def __ilshift__(self, x) -> Tensor: return self.assign(self.lshift(x))
  def __irshift__(self, x) -> Tensor: return self.assign(self.rshift(x))

  def __lt__(self, x) -> Tensor: return F.Less.apply(*self._broadcasted(x, False))
  def __gt__(self, x) -> Tensor: return F.Less.apply(*self._broadcasted(x, True))
  def __ge__(self, x) -> Tensor: return (self<x).logical_not()
  def __le__(self, x) -> Tensor: return (self>x).logical_not()
  def __eq__(self, x) -> Tensor: return F.Eq.apply(*self._broadcasted(x, True))       # type: ignore[override]
  def __ne__(self, x) -> Tensor: return (self==x).logical_not()                       # type: ignore[override]

  # ***** functional nn ops *****

  def linear(self, weight:Tensor, bias:Optional[Tensor]=None):
    """
    Applies a linear transformation to `self` using `weight` and `bias`.

    See: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2], [3, 4]])
    weight = Tensor([[1, 2], [3, 4]])
    bias = Tensor([1, 2])
    print(t.linear(weight, bias).numpy())
    ```
    """
    x = self.mul(weight) if len(weight.shape) == 1 else self.dot(weight)
    return x.add(bias) if bias is not None else x

  def sequential(self, ll:List[Callable[[Tensor], Tensor]]):
    """
    Applies a sequence of functions to `self` chaining the output of each function to the input of the next.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([1, 2, 3])
    print(t.sequential([lambda x: x * 2, lambda x: x + 1]).numpy())
    ```
    """
    return functools.reduce(lambda x,f: f(x), ll, self)

  def layernorm(self, axis=-1, eps:float=1e-5) -> Tensor:
    """
    Applies Layer Normalization over a mini-batch of inputs.

    - Described: https://paperswithcode.com/method/layer-normalization
    - Paper: https://arxiv.org/abs/1607.06450v1

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.randn(8, 10, 16) * 2 + 8
    print(t.mean().item(), t.std().item())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.layernorm()
    print(t.mean().item(), t.std().item())
    ```
    """
    y = (self - self.mean(axis, keepdim=True))
    return y.mul((y*y).mean(axis, keepdim=True).add(eps).rsqrt())

  def batchnorm(self, weight:Optional[Tensor], bias:Optional[Tensor], mean:Tensor, invstd:Tensor, axis:Union[int,Tuple[int,...]]=1) -> Tensor:
    """
    Applies Batch Normalization over a mini-batch of inputs.

    - Described: https://paperswithcode.com/method/batch-normalization
    - Paper: https://arxiv.org/abs/1502.03167

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.randn(8, 4, 16, 16) * 2 + 8
    print(t.mean().item(), t.std().item())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.batchnorm(None, None, t.mean(axis=(0,2,3)), t.var(axis=(0,2,3)).add(1e-5).rsqrt())
    print(t.mean().item(), t.std().item())
    ```
    """
    axis_ = argfix(axis)
    shape = tuple(s if ax in axis_ else 1 for ax, s in enumerate(self.shape))
    x = self - mean.reshape(shape)
    if weight is not None: x = x * weight.reshape(shape)
    ret = x.mul(invstd.reshape(shape) if len(invstd.shape) == len(axis_) else invstd)
    return (ret + bias.reshape(shape)) if bias is not None else ret

  def dropout(self, p=0.5) -> Tensor:
    """
    Applies dropout to `self`.

    NOTE: dropout is only applied when `Tensor.is_train_enabled` is `True`.

    - Described: https://paperswithcode.com/method/dropout
    - Paper: https://jmlr.org/papers/v15/srivastava14a.html

    ```python exec="true" source="above" session="tensor" result="python"
    Tensor.manual_seed(42)
    t = Tensor.randn(2, 2)
    with Tensor.train():
      print(t.dropout().numpy())
    ```
    """
    if not Tensor.is_train_enabled or p == 0: return self
    return self * (Tensor.rand(*self.shape, requires_grad=False, dtype=dtypes.default_float, device=self.device) >= p) * (1/(1.0 - p))

  def one_hot(self, num_classes:int) -> Tensor:
    """
    Converts `self` to a one-hot tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([0, 1, 3, 3, 4])
    print(t.one_hot(5).numpy())
    ```
    """
    return (self[..., None] == Tensor.arange(num_classes, requires_grad=False, device=self.device)).where(1, 0)

  def scaled_dot_product_attention(self, key:Tensor, value:Tensor, attn_mask:Optional[Tensor]=None,
                                   dropout_p:float=0.0, is_causal:bool=False) -> Tensor:
    """
    Computes scaled dot-product attention.
    `self` is the query tensor, `key` is the key tensor, and `value` is the value tensor.

    NOTE: it also works when `key` and `value` have symbolic shape.

    - Described: https://paperswithcode.com/method/scaled
    - Paper: https://arxiv.org/abs/1706.03762v7

    ```python exec="true" source="above" session="tensor" result="python"
    q = Tensor.randn(2, 4, 8)
    k = Tensor.randn(2, 4, 8)
    v = Tensor.randn(2, 4, 8)
    print(q.scaled_dot_product_attention(k, v).numpy())
    ```
    """
    assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
    if is_causal: attn_mask = Tensor.ones(self.shape[-2], key.shape[-2], requires_grad=False, device=self.device).tril(0).cast(dtypes.bool)
    if attn_mask is not None and attn_mask.dtype == dtypes.bool: attn_mask = (attn_mask == 0).where(-float("inf"), 0)
    qk = self @ key.transpose(-2,-1) / math.sqrt(self.shape[-1])
    return ((qk+attn_mask) if attn_mask is not None else qk).softmax(-1).dropout(dropout_p) @ value

  def binary_crossentropy(self, y:Tensor) -> Tensor:
    """
    Computes the binary cross-entropy loss between `self` and `y`.

    See: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([0.1, 0.9, 0.2])
    y = Tensor([0, 1, 0])
    print(t.binary_crossentropy(y).item())
    ```
    """
    return (-y*self.log() - (1-y)*(1-self).log()).mean()

  def binary_crossentropy_logits(self, y:Tensor) -> Tensor:
    """
    Computes the binary cross-entropy loss between `self` and `y` where `self` is logits.

    See: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2, -3])
    y = Tensor([0, 1, 0])
    print(t.binary_crossentropy_logits(y).item())
    ```
    """
    return (self.maximum(0) - y * self + (1 + self.abs().neg().exp()).log()).mean()

  def sparse_categorical_crossentropy(self, Y:Tensor, ignore_index=-1, label_smoothing=0.0) -> Tensor:
    """
    Computes the sparse categorical cross-entropy loss between `self` and `Y`.

    NOTE: `self` is logits and `Y` is the target labels.

    See: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[-1, 2, -3], [1, -2, 3]])
    Y = Tensor([1, 2])
    print(t.sparse_categorical_crossentropy(Y).item())
    ```
    """
    assert 0.0 <= label_smoothing <= 1.0, "label_smoothing must be in [0.0, 1.0]"
    log_probs, loss_mask = self.log_softmax(), (Y != ignore_index)
    y_counter = Tensor.arange(self.shape[-1], requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
    y = ((y_counter == Y.flatten().reshape(-1, 1)) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
    smoothing = label_smoothing * (log_probs.mean(-1) * loss_mask).sum()
    return -((1 - label_smoothing) * (log_probs * y).sum() + smoothing) / loss_mask.sum()

  # ***** cast ops *****

  def llvm_bf16_cast(self, dtype:DType):
    # hack for devices that don't support bfloat16
    assert self.dtype == dtypes.bfloat16
    return self.to("LLVM").bitcast(dtypes.uint16).cast(dtypes.uint32).mul(1<<16).bitcast(dtypes.float32).cast(dtype)
  def cast(self, dtype:DType) -> Tensor:
    """
    Casts `self` to a new dtype.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2.5, 3], dtype=dtypes.float)
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.cast(dtypes.int32)
    print(t.dtype, t.numpy())
    ```
    """
    return self if self.dtype == dtype else F.Cast.apply(self, dtype=dtype)
  def bitcast(self, dtype:DType) -> Tensor:
    """
    Bitcasts `self` to a new dtype of the same itemsize.

    `self` must not require a gradient.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2, 3], dtype=dtypes.int32)
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.bitcast(dtypes.uint32)
    print(t.dtype, t.numpy())
    ```
    """
    if self.requires_grad: raise RuntimeError("can't backprop through bitcast")
    return F.Cast.apply(self, dtype=dtype, bitcast=True) if self.dtype != dtype else self
  def float(self) -> Tensor:
    """
    Convenience method to cast `self` to a `float32` Tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2, 3], dtype=dtypes.int32)
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.float()
    print(t.dtype, t.numpy())
    ```
    """
    return self.cast(dtypes.float32)
  def half(self) -> Tensor:
    """
    Convenience method to cast `self` to a `float16` Tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, 2, 3], dtype=dtypes.int32)
    print(t.dtype, t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    t = t.half()
    print(t.dtype, t.numpy())
    ```
    """
    return self.cast(dtypes.float16)

  # ***** convenience stuff *****

  @property
  def ndim(self) -> int: return len(self.shape)
  def numel(self) -> sint: return prod(self.shape)
  def element_size(self) -> int: return self.dtype.itemsize
  def nbytes(self) -> int: return self.numel() * self.element_size()
  def is_floating_point(self) -> bool: return dtypes.is_float(self.dtype)
  def size(self, dim=None) -> Union[sint, Tuple[sint, ...]]: return self.shape if dim is None else self.shape[dim]

  # *** image Tensor function replacements ***

  def image_dot(self, w:Tensor, acc_dtype=None):
    # NOTE: we use a 1x1 conv2d to do the matmul. mxk @ kxn = (1,k,m,1).conv2d(n,k,1,1)
    n1, n2 = len(self.shape), len(w.shape)
    assert n1 != 0 and n2 != 0, f"both arguments to matmul need to be at least 1D, but they are {n1}D and {n2}D"
    assert self.shape[-1] == w.shape[-min(n2, 2)], f"Input Tensor shapes {self.shape} and {w.shape} cannot be multiplied ({self.shape[-1]} != {w.shape[-min(n2, 2)]})"  # noqa: E501
    bs, groups, cin, cout = prod(self.shape[0:-2]), prod(w.shape[0:-2]), w.shape[-2], w.shape[-1]
    out_shape_t = self.shape[0:-2] + (cout,-1) if len(self.shape) > 1 else (cout, )

    # NOTE: with NHWC we can remove the transposes
    # bs x groups*cin x H x W
    cx = self.transpose(self.ndim-1, self.ndim-2).reshape((bs//groups, groups*cin, -1, 1))
    # groups*cout x cin x H, W
    cw = w.transpose(w.ndim-1, w.ndim-2).reshape((groups*cout, cin, 1, 1))
    return cx.image_conv2d(cw, groups=groups, acc_dtype=acc_dtype).reshape(out_shape_t).transpose(self.ndim-1, self.ndim-2)

  def image_conv2d(self, weight:Tensor, bias:Optional[Tensor]=None, groups=1, stride=1, dilation=1, padding=0, acc_dtype=None):
    base_image_type = dtypes.imageh if getenv("FLOAT16", 0) else dtypes.imagef

    (bs,_,iy,ix), (cout,cin,H,W) = self.shape, weight.shape
    x, w = self, weight.reshape(groups, (rcout := cout//groups), cin, H, W)

    # hack for non multiples of 4 on cin
    if cin % 4 != 0 and not (cin == 1 and groups%4 == 0):
      x = x.reshape(bs, groups, cin, iy, ix)   # do this always?
      added_input_channels = 4 - (cin % 4)
      w = w.pad(tuple((0, added_input_channels) if i == 2 else None for i in range(w.ndim)))
      x = x.pad(tuple((0, added_input_channels) if i == 2 else None for i in range(x.ndim)))
      cin = cin + added_input_channels
      x = x.reshape(bs, groups*cin, iy, ix)

    # hack for non multiples of 4 on rcout
    added_output_channels = 0
    if rcout % 4 != 0 and not (rcout == 1 and groups%4 == 0):
      added_output_channels = 4 - (rcout % 4)
      rcout += added_output_channels
      cout = groups * rcout
      w = w.pad(tuple((0, added_output_channels) if i == 1 else None for i in range(w.ndim)))

    # packed (note: flipping bs and iy would make the auto-padding work)
    x = x.permute(0,2,3,1)
    cin_last = iy == 1 and ix == 1
    if cin == 1: w = w.reshape(cout//4,4,H,W).permute(0,2,3,1)
    elif cin_last: w = w.reshape(cout//4,4,cin//4,4,H,W).permute(0,4,2,5,1,3)
    else: w = w.reshape(cout//4,4,cin//4,4,H,W).permute(0,4,2,5,3,1)

    # contiguous creates the image, and early realize static weights (TODO: test for the static weight)
    if IMAGE >= 2: x,w = x.cast(base_image_type((bs*iy, ix*groups*cin//4, 4))), w.cast(base_image_type((cout//4, H*W*cin, 4)))
    x, w = x.contiguous(), w.contiguous()

    # expand out
    rcin_hi, rcin_lo = cin//4 if cin >= 4 else 1, 4 if cin >= 4 else 1
    cout_expand = [groups//4 if cin == 1 else groups, 4 if cin == 1 else 1, rcout//4 if rcout >= 4 else 1, 4 if rcout >= 4 else 1]
    x = x.reshape(bs, iy, ix, groups, rcin_hi, rcin_lo)
    if cin_last: w = w.reshape(cout//4, H, rcin_hi, W, 4, rcin_lo)
    else: w = w.reshape(cout//4, H, rcin_hi, W, rcin_lo, 4).permute(0,1,2,3,5,4)

    # padding
    padding_ = [padding]*4 if isinstance(padding, int) else (padding if len(padding) == 4 else [padding[1], padding[1], padding[0], padding[0]])
    x = x._slice((None, (-padding_[2], x.shape[1]+padding_[3]), (-padding_[0], x.shape[2]+padding_[1]), None, None, None))

    # prepare input
    x = x.permute(0,3,4,5,1,2)._pool((H, W), stride, dilation) # -> (bs, groups, rcin_hi, rcin_lo, oy, ox, H, W)
    x = x.permute(0,4,5,1,2,3,6,7).reshape(bs, (oy := x.shape[4]), (ox := x.shape[5]), *cout_expand[0:2], 1, 1, rcin_hi, rcin_lo, H, W)

    # prepare weights
    w = w.permute(0,4,2,5,1,3).reshape((1, 1, 1, *cout_expand, rcin_hi, rcin_lo, H, W))

    # the conv!
    ret = (x*w).cast(base_image_type((bs*oy, ox*cout//4, 4)) if IMAGE >= 2 else dtypes.float32).sum((-4, -3, -2, -1), acc_dtype=acc_dtype)

    # undo hack for non multiples of 4 on C.rcout
    if added_output_channels != 0:
      ret = ret.reshape(bs, oy, ox, groups, rcout)[:, :, :, :, :-added_output_channels]
      cout = groups * (rcout - added_output_channels)

    # NCHW output
    ret = ret.reshape(bs, oy, ox, cout).permute(0,3,1,2)
    return ret if bias is None else ret.add(bias.reshape(1, -1, 1, 1))

# register functions to move between devices
for device in Device._devices: setattr(Tensor, f"{device.lower()}", functools.partialmethod(Tensor.to, device))

if IMAGE:
  # if IMAGE>0 we install these replacement functions in Tensor (hack!)
  setattr(Tensor, "conv2d", Tensor.image_conv2d)
  setattr(Tensor, "dot", Tensor.image_dot)

# TODO: eventually remove this
def custom_random(out:Buffer):
  Tensor._seed += 1
  rng = np.random.default_rng(Tensor._seed)
  if out.dtype == dtypes.half: rng_np_buffer = (rng.integers(low=0, high=2047, size=out.size) / 2048).astype(np.half, copy=False)
  else: rng_np_buffer = rng.random(size=out.size, dtype=np.float32).astype(dtype=out.dtype.np, copy=False)
  out.copyin(rng_np_buffer.data)
