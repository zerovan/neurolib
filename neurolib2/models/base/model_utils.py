import jax.numpy as jnp


def compute_delay_matrix(fiber_length_matrix, signal_propagation_speed, segment_length=1.0):
    """
    Compute the delay matrix from the fiber length matrix and the signal
    velocity

        :param fiber_length_matrix:       A matrix containing the connection length in segment
        :param signal_propagation_speed:         Signal velocity in m/s
        :param segment_length:   Length of a single segment in mm

        :returns:    A matrix of connection delay in ms
    """

    normalized_length_matrix = fiber_length_matrix * segment_length
    if signal_propagation_speed > 0:
        delay_matrix = normalized_length_matrix / signal_propagation_speed  # Interareal delays in ms
    else:
        delay_matrix = fiber_length_matrix * 0.0
    delay_matrix = jnp.fill_diagonal(delay_matrix, 0.0, inplace=False)
    return delay_matrix
