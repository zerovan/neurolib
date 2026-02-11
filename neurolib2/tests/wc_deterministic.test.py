import sys, os
import numpy as np
import jax
import diffrax
import jax.numpy as jnp
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from neurolib2.models.wilsonCowanModel.wilsonCowanModel import WilsonCowan


def test_no_delay_no_noise_vs_neurolib():
    """
    Compare deterministic, single‑node Wilson‑Cowan dynamics against neurolib.
    Both models use identical parameters, zero inter‑areal coupling, and no noise.
    """
    from neurolib.models.wc import WCModel

    # ==================================================================
    # 1. Shared simulation settings
    # ==================================================================
    dt = 0.001               # time step [seconds]
    duration = 5.0           # total simulation time [seconds]
    num_steps = int(duration / dt) + 1
    t_eval = np.linspace(0.0, duration, num_steps)

    # Single node → no coupling
    fiber_count_matrix = jnp.zeros((1, 1), dtype=float)
    fiber_length_matrix = jnp.zeros((1, 1), dtype=float)

    # Initial state for our model: shape (4, 1)
    # [E, I, OU_exc, OU_inh]  (OU states are set to zero and won't evolve)
    E0 = 0.1
    I0 = 0.1
    initial_state = jnp.array([[E0], [I0], [0.0], [0.0]], dtype=float)

    # ==================================================================
    # 2. Shared model parameters (taken from typical neurolib examples)
    # ==================================================================
    # Time constants
    tau_e = 0.01    # 10 ms
    tau_i = 0.02    # 20 ms

    # Connection weights
    w_ee = 2.0
    w_ei = 2.0
    w_ie = 1.8
    w_ii = 0.5

    # Transfer function parameters
    a_e = 1.0
    a_i = 1.0
    theta_e = 1.0
    theta_i = 1.0

    # Baseline external input
    baseline_e = 1.0
    baseline_i = 0.0

    # ==================================================================
    # 3. neurolib model
    # ==================================================================
    neurolib_model = WCModel(
        Cmat=np.zeros((1, 1)),          # zero connectivity matrix
        Dmat=np.zeros((1, 1)),          # delay matrix (unused)
        dt=dt,
        duration=duration,
    )
    # Set parameters (neurolib uses plain numbers, not arrays)
    neurolib_model.params['tau_e'] = tau_e
    neurolib_model.params['tau_i'] = tau_i
    neurolib_model.params['w_ee'] = w_ee
    neurolib_model.params['w_ei'] = w_ei
    neurolib_model.params['w_ie'] = w_ie
    neurolib_model.params['w_ii'] = w_ii
    neurolib_model.params['a_e'] = a_e
    neurolib_model.params['a_i'] = a_i
    neurolib_model.params['theta_e'] = theta_e
    neurolib_model.params['theta_i'] = theta_i
    neurolib_model.params['baseline_e'] = baseline_e
    neurolib_model.params['baseline_i'] = baseline_i

    # Set initial condition (neurolib uses dictionaries)
    neurolib_model.initial_state = {'E': np.array([E0]), 'I': np.array([I0])}

    # Run simulation
    neurolib_model.run()

    # Extract results (time series are stored as 1D arrays for single node)
    neurolib_t = neurolib_model.t
    neurolib_E = neurolib_model.output['E']   # shape (num_steps,)
    neurolib_I = neurolib_model.output['I']   # shape (num_steps,)

    # ==================================================================
    # 4. Our JAX/Diffrax model
    # ==================================================================
    # Create model with zero coupling and zero delays
    jax_model = WilsonCowan(
        state=initial_state,
        dt=dt,
        fiber_count_matrix=fiber_count_matrix,
        fiber_length_matrix=fiber_length_matrix,
        seed=42,   # fixed seed for reproducibility (OU noise is disabled later)
    )

    # Overwrite default parameters with the test set
    jax_model.tau_e = jnp.array(tau_e)
    jax_model.tau_i = jnp.array(tau_i)
    jax_model.w_ee = jnp.array(w_ee)
    jax_model.w_ei = jnp.array(w_ei)
    jax_model.w_ie = jnp.array(w_ie)
    jax_model.w_ii = jnp.array(w_ii)
    jax_model.a_e = jnp.array(a_e)
    jax_model.a_i = jnp.array(a_i)
    jax_model.theta_e = jnp.array(theta_e)
    jax_model.theta_i = jnp.array(theta_i)
    jax_model.exc_ext_baseline = jnp.array(baseline_e)
    jax_model.inh_ext_baseline = jnp.array(baseline_i)

    # Disable the Ornstein‑Uhlenbeck noise process completely
    jax_model.sigma_ou = jnp.array(0.0)
    jax_model.tau_ou = jnp.array(0.0)
    jax_model.mean_exc_ou = jnp.array(0.0)
    jax_model.mean_inh_ou = jnp.array(0.0)

    # Force the model to use a fixed‑step Euler scheme to match neurolib.
    # We achieve this by overriding the `get_term` method to return only an
    # `ODETerm` (no diffusion) and by passing a constant‑step controller.
    # To avoid modifying the class, we temporarily monkey‑patch the method.
    original_get_term = jax_model.get_term

    def deterministic_get_term(ts):
        # Only the deterministic drift, no diffusion
        return diffrax.ODETerm(jax_model.dynamics)

    jax_model.get_term = deterministic_get_term.__get__(jax_model, WilsonCowan)

    # Also need to patch `simulate` to use Euler and constant steps.
    # We create a local wrapper.
    def deterministic_simulate(model, duration):
        t0, t1 = 0.0, duration
        ts = jnp.arange(t0, t1, model.dt)

        term = model.get_term(ts)   # now returns ODETerm
        solver = diffrax.Euler()
        stepsize_controller = diffrax.ConstantStepSize()

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=model.dt,
            y0=lambda t: model.history_fn(t),  # returns zeros
            args=None,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=stepsize_controller,
            delays=None,   # no delays for single node
            max_steps=16**4,
        )
        return sol.ts, jnp.array(sol.ys)

    # Run simulation (first call compiles; second call is fast)
    jax_ts, jax_states = deterministic_simulate(jax_model, duration)
    # Ensure compilation is complete
    jax.block_until_ready(jax_states)

    # Extract E and I traces: states shape (T, 4, 1) → (T,)
    jax_E = jax_states[:, 0, 0]
    jax_I = jax_states[:, 1, 0]

    # Restore original methods (good practice, though not strictly needed)
    jax_model.get_term = original_get_term

    # ==================================================================
    # 5. Comparison
    # ==================================================================
    # Interpolate neurolib results onto our exact time grid if necessary.
    # Both simulations use the same dt and start at 0, so time arrays should match.
    # But to be safe we align them via linear interpolation.
    from scipy.interpolate import interp1d

    neurolib_E_interp = interp1d(neurolib_t, neurolib_E, kind='linear',
                                 fill_value='extrapolate')(jax_ts)
    neurolib_I_interp = interp1d(neurolib_t, neurolib_I, kind='linear',
                                 fill_value='extrapolate')(jax_ts)

    # Compare with reasonable tolerances
    rtol = 1e-3
    atol = 1e-4

    assert jnp.allclose(jax_E, neurolib_E_interp, rtol=rtol, atol=atol), \
        "Excitatory activity mismatch"
    assert jnp.allclose(jax_I, neurolib_I_interp, rtol=rtol, atol=atol), \
        "Inhibitory activity mismatch"

    # (Optional) Plot differences if the test fails – helpful for debugging
    # plt.figure()
    # plt.plot(jax_ts, jax_E - neurolib_E_interp, label='E difference')
    # plt.plot(jax_ts, jax_I - neurolib_I_interp, label='I difference')
    # plt.legend()
    # plt.show()