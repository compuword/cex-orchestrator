"""
cex_gpu_protocol.py -- CXNP wire protocol for GPU inference.

Identical to cex_npu_protocol.py -- same binary format.
GPU server listens on port 7478 (NPU server listens on 7474).
Sharing the protocol means NPU and GPU servers are interchangeable
at the wire level; the orchestrator picks the backend.
"""

import struct
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

MAGIC            = b"CXNP"
HEADER_SIZE      = 12
CXNP_MAGIC       = MAGIC        # alias used by gpu_server / npu_proxy
CXNP_HEADER_SIZE = HEADER_SIZE  # alias used by gpu_server / backend_router

# Inference ports
DEFAULT_NPU_RX_PORT = 7474
DEFAULT_GPU_RX_PORT = 7478

# Health ports
DEFAULT_NPU_HEALTH_PORT = 7476
DEFAULT_GPU_HEALTH_PORT = 7479


class MsgType(IntEnum):
    INFER_REQUEST    = 0x10
    INFER_RESPONSE   = 0x11
    HEARTBEAT        = 0x20
    HEARTBEAT_ACK    = 0x21
    CAPABILITIES_REQ = 0x30
    CAPABILITIES_RESP= 0x31
    ERROR            = 0xFF


class DType(IntEnum):
    FLOAT32 = 0
    INT64   = 1
    UINT8   = 2
    INT8    = 3
    FLOAT16 = 4
    INT32   = 5
    BOOL    = 6


CXNP_HEADER_SIZE = HEADER_SIZE

_DTYPE_MAP = {
    DType.FLOAT32: np.float32,
    DType.INT64:   np.int64,
    DType.UINT8:   np.uint8,
    DType.INT8:    np.int8,
    DType.FLOAT16: np.float16,
    DType.INT32:   np.int32,
    DType.BOOL:    np.bool_,
}

_NP_TO_DTYPE = {v: k for k, v in _DTYPE_MAP.items()}


def _np_dtype(arr: np.ndarray) -> DType:
    for np_t, dt in _NP_TO_DTYPE.items():
        if arr.dtype == np_t:
            return dt
    return DType.FLOAT32


def encode_header(msg_type: int, payload_len: int) -> bytes:
    return MAGIC + struct.pack("<II", msg_type, payload_len)


def decode_header(raw: bytes) -> Tuple[int, int]:
    if raw[:4] != MAGIC:
        raise ValueError(f"Bad magic: {raw[:4]!r}")
    msg_type, payload_len = struct.unpack("<II", raw[4:12])
    return msg_type, payload_len


def encode_tensors(tensors: Dict[str, np.ndarray]) -> bytes:
    parts = [struct.pack("<I", len(tensors))]
    for name, arr in tensors.items():
        name_b = name.encode("utf-8")
        dtype  = _np_dtype(arr)
        raw    = arr.tobytes()
        parts.append(struct.pack("<I", len(name_b)))
        parts.append(name_b)
        parts.append(struct.pack("<B", int(dtype)))
        parts.append(struct.pack("<I", arr.ndim))
        parts.append(struct.pack(f"<{arr.ndim}Q", *arr.shape))
        parts.append(struct.pack("<I", len(raw)))
        parts.append(raw)
    return b"".join(parts)


def decode_tensors(buf: bytes, offset: int = 0) -> Tuple[Dict[str, np.ndarray], int]:
    count = struct.unpack_from("<I", buf, offset)[0]; offset += 4
    tensors: Dict[str, np.ndarray] = {}
    for _ in range(count):
        nlen  = struct.unpack_from("<I", buf, offset)[0]; offset += 4
        name  = buf[offset:offset + nlen].decode("utf-8"); offset += nlen
        dtype = DType(struct.unpack_from("<B", buf, offset)[0]); offset += 1
        ndim  = struct.unpack_from("<I", buf, offset)[0]; offset += 4
        shape = struct.unpack_from(f"<{ndim}Q", buf, offset); offset += ndim * 8
        dlen  = struct.unpack_from("<I", buf, offset)[0]; offset += 4
        raw   = buf[offset:offset + dlen]; offset += dlen
        tensors[name] = np.frombuffer(raw, dtype=_DTYPE_MAP[dtype]).reshape(shape)
    return tensors, offset


def build_infer_request(
    request_id: str,
    model_bytes: bytes,
    input_tensors: Dict[str, np.ndarray],
    output_names: Optional[List[str]] = None,
) -> bytes:
    rid_b    = request_id.encode("utf-8")
    names_b  = b"".join(
        struct.pack("<I", len(n.encode())) + n.encode()
        for n in (output_names or [])
    )
    tensor_b = encode_tensors(input_tensors)
    payload  = (
        struct.pack("<I", len(rid_b)) + rid_b
        + struct.pack("<I", len(model_bytes)) + model_bytes
        + struct.pack("<I", len(output_names or [])) + names_b
        + tensor_b
    )
    return encode_header(MsgType.INFER_REQUEST, len(payload)) + payload


def parse_infer_request(payload: bytes) -> dict:
    off   = 0
    rlen  = struct.unpack_from("<I", payload, off)[0]; off += 4
    rid   = payload[off:off + rlen].decode(); off += rlen
    mlen  = struct.unpack_from("<I", payload, off)[0]; off += 4
    model = payload[off:off + mlen]; off += mlen
    nout  = struct.unpack_from("<I", payload, off)[0]; off += 4
    names = []
    for _ in range(nout):
        nlen = struct.unpack_from("<I", payload, off)[0]; off += 4
        names.append(payload[off:off + nlen].decode()); off += nlen
    tensors, _ = decode_tensors(payload, off)
    return {"request_id": rid, "model_bytes": model, "output_names": names, "input_tensors": tensors}


def build_infer_response(
    request_id: str,
    output_tensors: Dict[str, np.ndarray],
    latency_ms: float,
) -> bytes:
    rid_b   = request_id.encode("utf-8")
    tensor_b= encode_tensors(output_tensors)
    payload = (
        struct.pack("<I", len(rid_b)) + rid_b
        + struct.pack("<f", latency_ms)
        + tensor_b
    )
    return encode_header(MsgType.INFER_RESPONSE, len(payload)) + payload


def parse_infer_response(payload: bytes) -> dict:
    off   = 0
    rlen  = struct.unpack_from("<I", payload, off)[0]; off += 4
    rid   = payload[off:off + rlen].decode(); off += rlen
    lat   = struct.unpack_from("<f", payload, off)[0]; off += 4
    tensors, _ = decode_tensors(payload, off)
    return {"request_id": rid, "latency_ms": lat, "output_tensors": tensors}


def build_heartbeat() -> bytes:
    return encode_header(MsgType.HEARTBEAT, 0)


def build_heartbeat_ack(npu_util_pct: float, queue_depth: int) -> bytes:
    payload = struct.pack("<fI", npu_util_pct, queue_depth)
    return encode_header(MsgType.HEARTBEAT_ACK, len(payload)) + payload


def build_error(message: str) -> bytes:
    msg_b   = message.encode("utf-8")
    payload = struct.pack("<I", len(msg_b)) + msg_b
    return encode_header(MsgType.ERROR, len(payload)) + payload
