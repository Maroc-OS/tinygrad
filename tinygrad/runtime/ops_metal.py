import os, subprocess, pathlib
import Metal, Cocoa, libdispatch # type: ignore
from typing import List, Any
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage
from tinygrad.helpers import prod, getenv, DEBUG, DType
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawBufferMapped

METAL_XCODE = getenv("METAL_XCODE")

class _METAL:
  def __init__(self) -> None:
    self.mtl_buffers_in_flight: List[Any] = []
    self.device = Metal.MTLCreateSystemDefaultDevice()
    self.mtl_queue = self.device.newCommandQueue()
  # TODO: is there a better way to do this?
  def synchronize(self) -> None:
    for cbuf in self.mtl_buffers_in_flight: cbuf.waitUntilCompleted()
    self.mtl_buffers_in_flight.clear()
METAL = _METAL()

class RawMetalBuffer(RawBufferMapped):
  def __init__(self, size:int, dtype:DType) -> None:
    super().__init__(size, dtype, METAL.device.newBufferWithBytesNoCopy_length_options_deallocator_(None, size * dtype.itemsize, Metal.MTLResourceStorageModeShared, None))
  def __del__(self) -> None:
    self.release()
    super().__del__()
  def release(self) -> None:
    if self._buf is not None:
      purgeable_state = 2  # 2 represents the MTLPurgeableState.Empty state
      self._buf.setPurgeableState_(purgeable_state)
      self._buf = None
  def _buffer(self):
    METAL.synchronize()
    return self._buf.contents().as_buffer(self._buf.length())

def unwrap(x) -> Any:
  ret, err = x
  assert err is None, str(err)
  return ret

class MetalProgram:
  def __init__(self, name:str, prg:str) -> None:
    if METAL_XCODE:
      air = subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metal', '-x', 'metal', '-c', '-', '-o', '-'], input=prg.encode('utf-8'))
      # NOTE: if you run llvm-dis on "air" you can see the llvm bytecode
      lib = subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metallib', '-', '-o', '-'], input=air)
      data = libdispatch.dispatch_data_create(lib, len(lib), None, None)
      self.library = unwrap(METAL.device.newLibraryWithData_error_(data, None))
    else:
      options = Metal.MTLCompileOptions.alloc().init()
      self.library = unwrap(METAL.device.newLibraryWithSource_options_error_(prg, options, None))
    if self.library is None:
      raise RuntimeError(f"No library to compile:\n {prg}")
    self.fxn = self.library.newFunctionWithName_(name)
    # hacks to disassemble shader
    if DEBUG >= 5:
      desc = Metal.MTLComputePipelineDescriptor.alloc().init()
      desc.setComputeFunction_(self.fxn)
      arc = unwrap(METAL.device.newBinaryArchiveWithDescriptor_error_(Metal.MTLBinaryArchiveDescriptor.alloc().init(), None))
      unwrap(x=arc.addComputePipelineFunctionsWithDescriptor_error_(desc, None) or arc.newComputePipelineStateWithDescriptor_error_(desc, None))
      unwrap(arc.serializeToURL_error_(Cocoa.NSURL.URLWithString_("file:///tmp/shader.bin"), None))
      # clone https://github.com/dougallj/applegpu.git in tinygrad/disassemblers
      os.system(f"cd {pathlib.Path(__file__).parent.parent.parent}/disassemblers/applegpu && python3 compiler_explorer.py /tmp/shader.bin")
    self.pipeline_state = unwrap(METAL.device.newComputePipelineStateWithFunction_error_(self.fxn, None))

  def __call__(self, global_size:List[int], local_size:List[int], *bufs, wait=False):
    if not local_size: local_size = [32]
    local_size += [1] * (3-len(local_size))
    global_size += [1] * (3-len(global_size))

    assert prod(local_size) <= self.pipeline_state.maxTotalThreadsPerThreadgroup(), f"local size {local_size} bigger than {self.pipeline_state.maxTotalThreadsPerThreadgroup()} with exec width {self.pipeline_state.threadExecutionWidth()} memory length {self.pipeline_state.staticThreadgroupMemoryLength()}"
    command_buffer = METAL.mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(self.pipeline_state)
    for i,a in enumerate(bufs): encoder.setBuffer_offset_atIndex_(a._buf, 0, i)
    encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
    encoder.endEncoding()
    command_buffer.commit()
    if wait:
      command_buffer.waitUntilCompleted()
      return command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    else:
      METAL.mtl_buffers_in_flight.append(command_buffer)

class MetalCodegen(CStyleCodegen):
  lang = CStyleLanguage(
    kernel_prefix = "#include <metal_stdlib>\n#define int64 long\nusing namespace metal;\nkernel", buffer_prefix = "device ", smem_prefix = "threadgroup ",
    barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);", float4 = "float4",
    gid = [f"gid.{chr(120+i)}" for i in range(3)], lid = [f"lid.{chr(120+i)}" for i in range(3)],
    extra_args = ['uint3 gid [[thread_position_in_grid]]', 'ushort3 lid [[thread_position_in_threadgroup]]'])

MetalBuffer = Compiled(RawMetalBuffer, MetalCodegen, MetalProgram, METAL.synchronize)
