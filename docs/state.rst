Physical and Logical Qubits
===========================
QSim supports simulating logical qubits, including a two-qubit code (:class:`qsim.state.TwoQubitCode`) and
the Jordan-Farhi-Shor code (:class:`qsim.state.JordanFarhiShor`). Alternatively, simple physical qubits
can be simulated with :class:`qsim.state.State`.
Each class contains several helpful class attributes specific to the code it specifies:

:SX: Stores the logical Pauli :math:`\sigma^x_L` matrix
:SY: Stores the logical Pauli :math:`\sigma^y_L` matrix
:SZ: Stores the logical Pauli :math:`\sigma^z_L` matrix
:basis: Stores the logical computational basis
:n: Represents the number of physical qubits to a single logical qubit (:math:`n=1` for physical qubits)

The :class:`qsim.state.State` class can be used to represent a state as a
ket or a density matrix.

.. autoclass:: qsim.state.State
    :members:
    :show-inheritance:

.. autoclass:: qsim.state.TwoQubitCode
    :members:
    :show-inheritance:

.. autoclass:: qsim.state.JordanFarhiShor
    :members:
    :show-inheritance:
