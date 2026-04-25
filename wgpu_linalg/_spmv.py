from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse
import wgpu
import pygfx
import fastplotlib as fpl
from fastplotlib.graphics.features import TextureArray


_WGSL_PATH = Path(__file__).with_name("spmv_csr.wgsl")
_WORKGROUP_SIZE = 32


"""
Experimental support for compute shaders.
This should eventually be a more generic class, and move to wgpu.utils, or a new library.
See https://github.com/pygfx/wgpu-py/issues/704
"""

import time

from typing import Optional, Union
import pygfx as gfx

# compute shader from pygfx with uniform buffer support
# TODO: move this into wgpu/pygfx/new lib
# TODO: ability to concatenate multiple steps
class ComputeShader:
    """Abstraction for a compute shader.

    Parameters
    ----------
    wgsl : str
        The compute shader's code as WGSL.
    entry_point : str | None
        The name of the wgsl function that must be called.
        If the wgsl code has only one entry-point (a function marked with ``@compute``)
        this argument can be omitted.
    label : str | None
        The label for this shader. Used to set labels of underlying wgpu objects,
        and in debugging messages. If not set, use the entry_point.
    report_time : bool
        When set to True, will print the spent time to run the shader.
    """

    def __init__(
        self,
        wgsl,
        *,
        entry_point: Optional[str] = None,
        label: Optional[str] = None,
        report_time: bool = False,
    ):
        # Fixed
        self._wgsl = wgsl
        self._entry_point = entry_point
        self._label = label or entry_point or ""
        self._report_time = report_time

        # Things that can be changed via the API.
        # _resources maps index -> (object, clear, kind) where kind is "resource" or "uniform".
        self._resources = {}
        self._constants = {}

        # Flag to keep track whether this object changed.
        # Note that this says nothing about the contents of buffers/textures used as input.
        self._changed = True

        # Internal variables
        self._device = None
        self._shader_module = None
        self._pipeline = None
        self._bind_group = None

    @property
    def changed(self) -> bool:
        """Whether the shader has been changed.

        This can be a new value for a constant, or a different resource.
        Note that this says nothing about the values inside a buffer or texture resource.
        This value is reset when ``dispatch()`` is called.
        """
        return self._changed

    def set_resource(
        self,
        index: int,
        resource: Union[gfx.Buffer, gfx.Texture, wgpu.GPUBuffer, wgpu.GPUTexture],
        *,
        clear=False,
    ):
        """Set a resource.

        Parameters
        ----------
        index : int
            The binding index to connect this resource to. (The group is hardcoded to zero for now.)
        resource : buffer | texture
            The buffer or texture to attach. Can be a wgpu or pygfx resource.
        clear : bool
            When set to True (only possible for a buffer), the resource is cleared to zeros
            right before running the shader.
        """
        # Check
        if not isinstance(index, int):
            raise TypeError(f"ComputeShader resource index must be int, not {index!r}.")
        if not isinstance(
            resource, (gfx.Buffer, gfx.Texture, wgpu.GPUBuffer, wgpu.GPUTexture)
        ):
            raise TypeError(
                f"ComputeShader resource value must be gfx.Buffer, gfx.Texture, wgpu.GPUBuffer, or wgpu.GPUTexture, not {resource!r}"
            )
        clear = bool(clear)
        if clear and not isinstance(
            resource, (gfx.Buffer, gfx.Texture, wgpu.GPUBuffer)
        ):
            raise ValueError("Can only clear a buffer, not a texture.")

        # Reject collision with an existing uniform on the same index.
        old_value = self._resources.get(index)
        if old_value is not None and old_value[2] == "uniform":
            raise ValueError(
                f"Binding index {index} is already used as a uniform; choose a different index."
            )

        # Value to store
        new_value = (resource, bool(clear), "resource")

        # Update if different
        if new_value != old_value:
            if resource is None:
                self._resources.pop(index, None)
            else:
                self._resources[index] = new_value
            self._bind_group = None
            self._changed = True

    def set_uniform(self, index: int, data):
        """Set a uniform buffer binding.

        Uniform buffers are intended for small, frequently updated values such as
        per-frame scalars or shape metadata. Writing a uniform with the same
        byte-size as the previous write does not invalidate the pipeline or bind
        group — only ``queue.write_buffer`` is called. Resizing the uniform (or
        assigning to a previously unused index) rebuilds the bind group.

        Parameters
        ----------
        index : int
            The binding index. Group is hardcoded to zero, like ``set_resource``.
        data : numpy.ndarray | bytes | bytearray | memoryview
            The bytes to upload. Size must be a multiple of 4 bytes. The
            underlying buffer is sized to the next multiple of 16 bytes (WGSL
            uniform layout rule).
        """
        if not isinstance(index, int):
            raise TypeError(f"ComputeShader uniform index must be int, not {index!r}.")

        if isinstance(data, np.ndarray):
            payload = np.ascontiguousarray(data)
        elif isinstance(data, (bytes, bytearray, memoryview)):
            payload = np.frombuffer(data, dtype=np.uint8).copy()
        else:
            raise TypeError(
                f"ComputeShader uniform value must be numpy.ndarray or bytes-like, not {data!r}."
            )

        nbytes = payload.nbytes
        if nbytes == 0:
            raise ValueError("ComputeShader uniform data must not be empty.")
        if nbytes % 4 != 0:
            raise ValueError(
                f"ComputeShader uniform data size {nbytes} must be a multiple of 4 bytes."
            )
        padded_size = (nbytes + 15) & ~15  # WGSL uniform layout requires 16-byte alignment

        old_value = self._resources.get(index)
        if old_value is not None and old_value[2] != "uniform":
            raise ValueError(
                f"Binding index {index} is already used as a storage resource; choose a different index."
            )

        # Allocate (or reallocate) the uniform buffer on first use or on size change.
        if old_value is None or old_value[0].size != padded_size:
            if self._device is None:
                self._device = gfx.renderers.wgpu.Shared.get_instance().device
            buffer = self._device.create_buffer(
                label=f"{self._label} uniform[{index}]",
                size=padded_size,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            self._resources[index] = (buffer, False, "uniform")
            self._bind_group = None
            self._changed = True
        else:
            buffer = old_value[0]

        self._device.queue.write_buffer(buffer, 0, payload)

    def set_constant(self, name: str, value: Union[bool, int, float, None]):
        """Set override constant.

        Setting override constants don't require shader recompilation, but does
        require re-creating the pipeline object. So it's less suited for things
        that change on every draw, use ``set_uniform`` for those.
        """
        # Check
        if not isinstance(name, str):
            raise TypeError(f"ComputeShader constant name must be str, not {name!r}.")
        if not (value is None or isinstance(value, (bool, int, float, ))):
            raise TypeError(
                f"ComputeShader constant value must be bool, int, float, or None, not {value!r}."
            )

        # Update if different
        old_value = self._constants.get(name)
        if value != old_value:
            if value is None:
                self._constants.pop(name, None)
            else:
                self._constants[name] = value
            self._pipeline = None
            self._changed = True

    def _get_native_resource(self, resource):
        if isinstance(resource, gfx.Resource):
            return gfx.renderers.wgpu.engine.update.ensure_wgpu_object(resource)
        return resource

    def _get_bindings_from_resources(self):
        bindings = []
        for index, (resource, _, _) in self._resources.items():
            # Get wgpu.GPUBuffer or wgpu.GPUTexture
            wgpu_object = self._get_native_resource(resource)
            if isinstance(wgpu_object, wgpu.GPUBuffer):
                bindings.append(
                    {
                        "binding": index,
                        "resource": {
                            "buffer": wgpu_object,
                            "offset": 0,
                            "size": wgpu_object.size,
                        },
                    }
                )
            elif isinstance(wgpu_object, wgpu.GPUTexture):
                bindings.append(
                    {
                        "binding": index,
                        "resource": wgpu_object.create_view(
                            usage=wgpu.TextureUsage.STORAGE_BINDING
                        ),
                    }
                )
            else:
                raise RuntimeError(f"Unexpected resource: {resource}")
        return bindings

    def dispatch(self, nx, ny=1, nz=1):
        """Dispatch the workgroups, i.e. run the shader."""
        nx, ny, nz = int(nx), int(ny), int(nz)

        # Reset
        self._changed = False

        # Get device
        if self._device is None:
            self._shader_module = None
            self._device = gfx.renderers.wgpu.Shared.get_instance().device
        device = self._device

        # Compile the shader
        if self._shader_module is None:
            self._pipeline = None
            self._shader_module = device.create_shader_module(
                label=self._label, code=self._wgsl
            )

        # Get the pipeline object
        if self._pipeline is None:
            self._bind_group = None
            self._pipeline = device.create_compute_pipeline(
                label=self._label,
                layout="auto",
                compute={
                    "module": self._shader_module,
                    "entry_point": self._entry_point,
                    "constants": self._constants,
                },
            )

        # Get the bind group object
        if self._bind_group is None:
            bind_group_layout = self._pipeline.get_bind_group_layout(0)
            bindings = self._get_bindings_from_resources()
            self._bind_group = device.create_bind_group(
                label=self._label, layout=bind_group_layout, entries=bindings
            )

        # Make sure that all used resources have a wgpu-representation, and are synced
        for resource, _, _ in self._resources.values():
            if isinstance(resource, gfx.Resource):
                gfx.renderers.wgpu.engine.update.update_resource(resource)

        t0 = time.perf_counter()

        # Start!
        command_encoder = device.create_command_encoder(label=self._label)

        # Maybe clear some buffers
        for resource, clear, _ in self._resources.values():
            if clear:
                command_encoder.clear_buffer(self._get_native_resource(resource))

        # Do the compute pass
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self._pipeline)
        compute_pass.set_bind_group(0, self._bind_group)
        compute_pass.dispatch_workgroups(nx, ny, nz)
        compute_pass.end()

        # Submit!
        device.queue.submit([command_encoder.finish()])

        # Timeit
        if self._report_time:
            device._poll_wait()  # wait for the GPU to finish
            t1 = time.perf_counter()
            what = f"Computing {self._label!r}" if self._label else "Computing"
            print(f"{what} took {(t1 - t0) * 1000:0.1f} ms")



@dataclass(frozen=True, slots=True)
class _CsrBuffers:
    """The three GPU storage buffers that represent a CSR matrix."""
    indptr: wgpu.GPUBuffer
    indices: wgpu.GPUBuffer
    values: wgpu.GPUBuffer


class SparseDenseImage:
    """Live image of ``y = scale_factor * (A @ C[:, t]) + scale_add``, reshaped to ``(m, n)``.

    A is sparse [p, k] in CSR. C is dense [k, T], float32. The output column
    vector of length p is interpreted row-major as an (m, n) image and shown
    via a fastplotlib :class:`ImageGraphic` whose texture is written by a
    compute shader, so the data never leaves the GPU.

    Parameters
    ----------
    A : scipy.sparse CSR (csr_matrix or csr_array), float32
        Sparse matrix of shape ``(p, k)``.
    C : numpy.ndarray, float32
        Dense matrix of shape ``(k, T)``.
    shape : (int, int)
        ``(m, n)`` with ``m * n == p``.
    scale_factor : numpy.ndarray of shape (p,), float32, optional
        Per-row multiplicative gain applied to ``A @ C[:, t]``. Defaults to
        all ones.
    scale_add : numpy.ndarray of shape (p,), float32, optional
        Per-row additive bias applied after the scale_factor multiply.
        Defaults to all zeros.
    **image_kwargs
        Forwarded to :class:`fastplotlib.ImageGraphic` (e.g. ``vmin``,
        ``vmax``, ``cmap``).
    """

    def __init__(self, A, C, shape, scale_factor=None, scale_add=None, **image_kwargs):
        if not (scipy.sparse.issparse(A) and A.format == "csr"):
            raise TypeError(
                f"A must be scipy.sparse CSR, got {type(A).__name__} "
                f"(format={getattr(A, 'format', None)!r}). Call A.tocsr() first."
            )
        if A.dtype != np.float32:
            raise TypeError(f"A.dtype must be float32, got {A.dtype}.")
        if not isinstance(C, np.ndarray):
            raise TypeError(f"C must be a numpy.ndarray, got {type(C).__name__}.")
        if C.ndim != 2:
            raise ValueError(f"C must be 2D, got shape {C.shape}.")
        if C.dtype != np.float32:
            raise TypeError(f"C.dtype must be float32, got {C.dtype}.")

        p, k = A.shape
        k_C, T = C.shape
        if k != k_C:
            raise ValueError(f"A.shape[1] ({k}) != C.shape[0] ({k_C}).")

        if not (isinstance(shape, tuple) and len(shape) == 2):
            raise TypeError(f"shape must be a (m, n) tuple, got {shape!r}.")
        m, n = shape
        if not (isinstance(m, int) and isinstance(n, int)):
            raise TypeError(
                f"shape entries must be int, got ({type(m).__name__}, {type(n).__name__})."
            )
        if m * n != p:
            raise ValueError(f"m*n ({m}*{n}={m*n}) must equal A.shape[0] ({p}).")

        scale_factor = _check_scale(scale_factor, p, "scale_factor", default_value=1.0)
        scale_add = _check_scale(scale_add, p, "scale_add", default_value=0.0)

        device = pygfx.renderers.wgpu.get_shared().device
        max_dim = device.limits["max-texture-dimension-2d"]
        if m > max_dim or n > max_dim:
            raise ValueError(
                f"shape ({m}, {n}) exceeds the device 2D texture limit ({max_dim})."
            )

        self._p, self._T = p, T
        self._m, self._n = m, n

        self._A = _CsrBuffers(
            indptr=_storage_buffer(device, np.ascontiguousarray(A.indptr, np.uint32), "indptr"),
            indices=_storage_buffer(device, np.ascontiguousarray(A.indices, np.uint32), "indices"),
            values=_storage_buffer(device, np.ascontiguousarray(A.data, np.float32), "values"),
        )
        self._C = _storage_buffer(device, np.ascontiguousarray(C), "C")
        self._scale_factor = _storage_buffer(device, scale_factor, "scale_factor")
        self._scale_add = _storage_buffer(device, scale_add, "scale_add")

        # Output texture, GPU-only, writable by the compute shader and sampled
        # by the renderer. fastplotlib's TextureArray creates the pygfx.Texture
        # with COPY_DST | usage and applies the seed via send_data.
        seed = np.zeros((m, n), dtype=np.float32)
        self._texture_array = TextureArray(
            seed,
            cpu_buffer=False,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.COPY_SRC
            ),
        )
        self._out_texture = self._texture_array.buffer[0, 0]

        # Override constants are baked into the pipeline at first dispatch.
        # Per-frame `t` lives in a uniform that we rewrite on each setter call.
        self._compute = ComputeShader(
            _WGSL_PATH.read_text(),
            entry_point="spmv",
            label="wgpu_linalg.spmv_csr",
        )
        self._compute.set_constant("wg_size", _WORKGROUP_SIZE)
        self._compute.set_constant("T", T)
        self._compute.set_constant("n_cols", n)
        self._compute.set_resource(0, self._A.indptr)
        self._compute.set_resource(1, self._A.indices)
        self._compute.set_resource(2, self._A.values)
        self._compute.set_resource(3, self._C)
        self._compute.set_resource(5, self._out_texture)
        self._compute.set_resource(6, self._scale_factor)
        self._compute.set_resource(7, self._scale_add)

        # Auto-estimate vmin/vmax from sampled frames unless the caller passed them.
        if "vmin" not in image_kwargs or "vmax" not in image_kwargs:
            est_vmin, est_vmax = self._estimate_clim()
            image_kwargs.setdefault("vmin", est_vmin)
            image_kwargs.setdefault("vmax", est_vmax)
        self._image_graphic = fpl.ImageGraphic(self._texture_array, **image_kwargs)

        self.t = 0

    @property
    def image_graphic(self) -> fpl.ImageGraphic:
        return self._image_graphic

    def _estimate_clim(self) -> tuple[float, float]:
        """Estimate (vmin, vmax) by sampling 10 equally spaced frames.

        Samples ``t`` in ``[1, T-1]`` (skipping 0, which is often a degenerate
        transient that would collapse the colormap range) and returns the
        overall (min, max) across those frames. Falls back to ``(0.0, 1.0)``
        when ``T <= 1``.
        """
        n_samples = min(10, self._T - 1)
        if n_samples < 1:
            return 0.0, 1.0
        sample_ts = np.unique(
            np.linspace(1, self._T - 1, n_samples).round().astype(np.int64)
        )
        vmin = np.inf
        vmax = -np.inf
        for t in sample_ts:
            self.t = int(t)
            frame = self.to_numpy()
            vmin = min(vmin, float(frame.min()))
            vmax = max(vmax, float(frame.max())) / 2
        return vmin, vmax

    def to_numpy(self) -> np.ndarray:
        """Download the current texture contents as an ``(m, n)`` float32 array.

        Reads back the result of the most recent ``t`` setter dispatch. This
        triggers a GPU sync (the texture-to-buffer copy is queued after the
        compute dispatch and the call blocks until the data is mappable).
        """
        device = pygfx.renderers.wgpu.get_shared().device
        wgpu_texture = pygfx.renderers.wgpu.engine.update.ensure_wgpu_object(self._out_texture)
        raw = device.queue.read_texture(
            source={"texture": wgpu_texture, "origin": (0, 0, 0), "mip_level": 0},
            data_layout={"offset": 0, "bytes_per_row": self._n * 4},
            size=(self._n, self._m, 1),
        )
        return np.frombuffer(raw, dtype=np.float32).reshape(self._m, self._n).copy()

    @property
    def t(self) -> int:
        return self._t

    @t.setter
    def t(self, value):
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"t must be int, got {type(value).__name__}.")
        v = int(value)
        if v < 0 or v >= self._T:
            raise IndexError(f"t={v} out of range [0, {self._T}).")
        self._t = v
        self._compute.set_uniform(4, np.array([v], dtype=np.uint32))
        self._compute.dispatch((self._p + _WORKGROUP_SIZE - 1) // _WORKGROUP_SIZE)


def _storage_buffer(device, array: np.ndarray, label: str) -> wgpu.GPUBuffer:
    buf = device.create_buffer(
        label=f"wgpu_linalg.{label}",
        size=array.nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    device.queue.write_buffer(buf, 0, array)
    return buf


def _check_scale(arr, p: int, name: str, default_value: float) -> np.ndarray:
    if arr is None:
        return np.full(p, default_value, dtype=np.float32)
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr).__name__}.")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}.")
    if arr.shape[0] != p:
        raise ValueError(f"{name} length {arr.shape[0]} must equal p ({p}).")
    if arr.dtype != np.float32:
        raise TypeError(f"{name}.dtype must be float32, got {arr.dtype}.")
    return np.ascontiguousarray(arr)
