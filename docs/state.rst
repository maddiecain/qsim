Physical qudits
===============
The :class:`qsim.state.State` class can be used to represent an ensemble of qudits as a
ket or a density matrix.

.. autoclass:: qsim.state.State
    :members:
    :show-inheritance:

Logical qubits
===============
QSim supports simulating logical qubits, including a two-qubit code (:class:`qsim.state.TwoQubitCode`) and
the Jordan-Farhi-Shor code (:class:`qsim.state.JordanFarhiShor`).
Each class contains several class attributes specific to the code it specifies:

:X: Stores the logical Pauli :math:`\sigma^x_L` matrix
:Y: Stores the logical Pauli :math:`\sigma^y_L` matrix
:Z: Stores the logical Pauli :math:`\sigma^z_L` matrix
:basis: Stores the logical computational basis states
:n: Represents the number of physical qubits to a single logical qubit (:math:`n=1` for physical qudits)

.. autoclass:: qsim.state.TwoQubitCode
    :members:
    :show-inheritance:

.. autoclass:: qsim.state.ThreeQubitCode
    :members:
    :show-inheritance:

.. autoclass:: qsim.state.JordanFarhiShor
    :members:
    :show-inheritance:

.. autoclass:: qsim.state.ETThreeQubitCode
    :members:
    :show-inheritance: