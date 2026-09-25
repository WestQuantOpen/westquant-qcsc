"""Resource model — QPU backend capabilities and local compute resources."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResourceCertainty(str, Enum):
    KNOWN = "known"
    REPORTED = "reported"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


@dataclass
class QPUBackend:
    backend_id: str
    provider: str = "unknown"
    qubits: int = 0
    connectivity: str = "unknown"
    native_gates: list[str] = field(default_factory=list)
    parallel_gate_support: bool = False
    measurement_capabilities: str = "unknown"
    max_shots: int = 0
    queue_model: str = "unknown"
    pricing_model: str = "unknown"
    known_latency: str = "unknown"
    certainty: ResourceCertainty = ResourceCertainty.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "provider": self.provider,
            "qubits": self.qubits,
            "connectivity": self.connectivity,
            "native_gates": self.native_gates,
            "parallel_gate_support": self.parallel_gate_support,
            "measurement_capabilities": self.measurement_capabilities,
            "max_shots": self.max_shots,
            "queue_model": self.queue_model,
            "pricing_model": self.pricing_model,
            "known_latency": self.known_latency,
            "certainty": self.certainty.value,
            "metadata": self.metadata,
        }


# Known backends
KNOWN_BACKENDS: dict[str, QPUBackend] = {
    "ibm_kyiv": QPUBackend(
        backend_id="ibm_kyiv", provider="IBM Quantum", qubits=127,
        connectivity="heavy_hex", parallel_gate_support=True,
        max_shots=100000, certainty=ResourceCertainty.REPORTED,
    ),
    "ibm_brisbane": QPUBackend(
        backend_id="ibm_brisbane", provider="IBM Quantum", qubits=127,
        connectivity="heavy_hex", parallel_gate_support=True,
        max_shots=100000, certainty=ResourceCertainty.REPORTED,
    ),
    "ibm_sherbrooke": QPUBackend(
        backend_id="ibm_sherbrooke", provider="IBM Quantum", qubits=127,
        connectivity="heavy_hex", parallel_gate_support=True,
        max_shots=100000, certainty=ResourceCertainty.REPORTED,
    ),
    "fake_provider": QPUBackend(
        backend_id="fake_provider", provider="Qiskit Fake Provider",
        qubits=0, certainty=ResourceCertainty.KNOWN,
        metadata={"note": "Simulator — not a real QPU"},
    ),
    "aer_simulator": QPUBackend(
        backend_id="aer_simulator", provider="Qiskit Aer",
        qubits=0, certainty=ResourceCertainty.KNOWN,
        metadata={"note": "Local simulator — not a QPU"},
    ),
}


def is_qpu_backend(backend_id: str) -> bool:
    """Check if a backend ID refers to a real QPU vs a simulator."""
    sim_keywords = ["simulator", "sim", "fake", "aer", "statevector", "mps", "stabilizer"]
    lower = backend_id.lower()
    for kw in sim_keywords:
        if kw in lower:
            return False
    return True


def get_backend(backend_id: str) -> QPUBackend | None:
    if backend_id in KNOWN_BACKENDS:
        return KNOWN_BACKENDS[backend_id]
    # Unknown backend — return a stub
    return QPUBackend(
        backend_id=backend_id,
        certainty=ResourceCertainty.UNKNOWN,
    )


@dataclass
class LocalResourceModel:
    cpu_cores: int = 0
    ram_gb: float = 0.0
    gpu_type: str | None = None
    gpu_memory_gb: float = 0.0
    has_gpu: bool = False
    detected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "ram_gb": self.ram_gb,
            "gpu_type": self.gpu_type,
            "gpu_memory_gb": self.gpu_memory_gb,
            "has_gpu": self.has_gpu,
            "detected": self.detected,
        }


def detect_local_resources() -> LocalResourceModel:
    """Detect local compute resources. Only called with user consent."""
    import multiprocessing
    import psutil  # type: ignore

    model = LocalResourceModel()
    model.detected = True
    try:
        model.cpu_cores = multiprocessing.cpu_count()
    except Exception:
        model.cpu_cores = 0
    try:
        model.ram_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:
        pass
    # GPU detection is platform-specific; skip for v0.1
    return model


# Simulation feasibility thresholds
STATEVECTOR_MAX_QUBITS_CPU = 28  # ~16GB for complex128
STATEVECTOR_MAX_QUBITS_GPU = 30  # ~32GB for complex128 on GPU
STABILIZER_MAX_QUBITS = 100  # Clifford circuits
MPS_MAX_QUBITS = 50  # Depends on entanglement structure


def statevector_memory_gb(n_qubits: int) -> float:
    """Estimate statevector memory in GB (complex128 = 16 bytes per amplitude)."""
    return 16 * (2 ** n_qubits) / (1024**3)


def is_statevector_feasible(n_qubits: int, local: LocalResourceModel | None = None) -> bool:
    mem = statevector_memory_gb(n_qubits)
    if local and local.detected:
        return mem < local.ram_gb
    return n_qubits <= STATEVECTOR_MAX_QUBITS_CPU


def is_stabilizer_feasible(circuit_info: dict[str, Any]) -> bool:
    """Check if circuit is Clifford-heavy (stabilizer-simulatable)."""
    non_clifford = circuit_info.get("non_clifford_gates", 0)
    return non_clifford == 0


def is_mps_feasible(circuit_info: dict[str, Any]) -> bool:
    """Heuristic check for MPS/tensor network suitability."""
    depth = circuit_info.get("depth", 0)
    n_qubits = circuit_info.get("n_qubits", 0)
    connectivity = circuit_info.get("connectivity", "unknown")
    # MPS works well for shallow circuits with limited entanglement
    if n_qubits == 0:
        return False
    if depth < 20 and n_qubits <= MPS_MAX_QUBITS:
        return True
    if connectivity in ("line", "ring") and depth < 50:
        return True
    return False
