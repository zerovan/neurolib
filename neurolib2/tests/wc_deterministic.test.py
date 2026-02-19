#!/usr/bin/env python3
"""
Deterministic validation of JAX/Diffrax Wilson‑Cowan against neurolib.
This script compares the time series of a single, uncoupled, noise‑free node.
"""

import sys
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import diffrax
from scipy.interpolate import interp1d
from neurolib.utils.collections import dotdict 

# ------------------------------------------------------------
# 1. Import your own Wilson‑Cowan model
#    (Adjust the path/import to match your actual project structure)
# ------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
try:
    # This import expects your model to be at:
    #   neurolib2/models/wilsonCowanModel/wilsonCowanModel.py
    from neurolib2.models.wilsonCowanModel.wilsonCowanModel import WilsonCowan
except ImportError:
    print("ERROR: Could not import your WilsonCowan model.")
    print("Please adjust the sys.path or import statement.")
    sys.exit(1)


# class DeterministicWilsonCowan(WilsonCowan):
#     def get_term(self, ts):
#         return diffrax.ODETerm(self.dynamics)
# ------------------------------------------------------------
# 2. Import neurolib for ground truth (skip test if not installed)
# ------------------------------------------------------------
try:
    from neurolib.models.wc import WCModel
    NEUROLIB_AVAILABLE = True
except ImportError:
    NEUROLIB_AVAILABLE = False
    print("WARNING: neurolib is not installed. Cannot perform comparison.")
    print("Install neurolib with: pip install neurolib")


def test_no_delay_no_noise_vs_neurolib():
    """Compare deterministic, single‑node Wilson‑Cowan dynamics against neurolib."""
    if not NEUROLIB_AVAILABLE:
        print("SKIPPED: neurolib not available")
        return

    print("\n" + "="*70)
    print("Deterministic Wilson‑Cowan validation (no delays, no noise)")
    print("="*70)

    # ==================================================================
    # 1. Shared simulation settings
    # ==================================================================
    dt = 0.001               # time step [seconds]
    duration = 5.0           # total simulation time [seconds]
    print(f"\nSimulation: dt = {dt} s, duration = {duration} s")
    print(f"Steps: {int(duration/dt)+1}")

    # Single node → no coupling
    fiber_count_matrix = jnp.zeros((1, 1), dtype=float)
    fiber_length_matrix = jnp.zeros((1, 1), dtype=float)

    # Initial state for your model: shape (4, 1) → [E, I, OU_exc, OU_inh]
    E0, I0 = 0.1, 0.1
    initial_state = jnp.array([[E0], [I0], [0.0], [0.0]], dtype=float)

    # ==================================================================
    # 2. Shared model parameters (typical neurolib values)
    # ==================================================================
    tau_e = 0.01    # 10 ms
    tau_i = 0.02    # 20 ms
    w_ee = 2.0
    w_ei = 2.0
    w_ie = 1.8
    w_ii = 0.5
    a_e = 1.0
    a_i = 1.0
    theta_e = 1.0
    theta_i = 1.0
    baseline_e = 1.0
    baseline_i = 0.0
    seed = 42

    # ==================================================================
    # 3. neurolib model (ground truth)
    # ==================================================================
    print("\n--- Running neurolib ---")

    params = {
        # ----- Runtime -----
        'dt': dt,
        'duration': duration,
        'seed': 42,

        # ----- Connectivity (single node) -----
        'Cmat': np.zeros((1, 1)),
        'lengthMat': np.zeros((1, 1)),
        'signalV': 20.0,
        'K_gl': 0.6,
        'N': 1,

        # ----- Local node parameters (neurolib names) -----
        'tau_exc': tau_e,
        'tau_inh': tau_i,
        'c_excexc': w_ee,
        'c_excinh': w_ei,
        'c_inhexc': w_ie,
        'c_inhinh': w_ii,
        'a_exc': a_e,
        'a_inh': a_i,
        'mu_exc': theta_e,
        'mu_inh': theta_i,
        'exc_ext_baseline': baseline_e,
        'inh_ext_baseline': baseline_i,

        # ----- External input (must exist) -----
        'exc_ext': 0.0,
        'inh_ext': 0.0,

        # ----- Noise (explicitly zero) -----
        'sigma_ou': 0.0,
        'tau_ou': 5.0,
        'exc_ou_mean': 0.0,
        'inh_ou_mean': 0.0,
        'exc_ou': np.zeros((1,)),
        'inh_ou': np.zeros((1,)),

        # ----- Initial conditions -----
        'exc_init': np.array([[E0]]),
        'inh_init': np.array([[I0]]),
    }

    # Convert to dotdict (required by neurolib)
    params = dotdict(params)

    # Diagnostic prints
    print(f"  params type: {type(params)}")
    print(f"  params.dt = {params.dt}, params.duration = {params.duration}")
    print(f"  params.Cmat shape: {params.Cmat.shape}, params.N = {params.N}")

    # Create model
    try:
        neurolib_model = WCModel(params=params)
        print("  ✅ WCModel created successfully")
    except Exception as e:
        print(f"  ❌ Failed to create WCModel: {e}")
        import traceback
        traceback.print_exc()
        return

    # Set initial state
    neurolib_model.initial_state = {'E': np.array([E0]), 'I': np.array([I0])}

    # Run simulation
    print("  Calling run()...")
    t0_neurolib = time.perf_counter()
    try:
        neurolib_model.run()
        print("  ✅ run() completed without exception")
    except Exception as e:
        print(f"  ❌ run() raised an exception: {e}")
        import traceback
        traceback.print_exc()
        return
    t_neurolib = time.perf_counter() - t0_neurolib

    # --- Access the FULL output dictionary (neurolib_model.outputs) ---
    print(f"  After run: type(neurolib_model.outputs) = {type(neurolib_model.outputs)}")
    if isinstance(neurolib_model.outputs, dict):
        print(f"  outputs keys: {list(neurolib_model.outputs.keys())}")
        neurolib_t = neurolib_model.t
        neurolib_E = neurolib_model.outputs['exc'].squeeze()   # ✅ correct
        neurolib_I = neurolib_model.outputs['inh'].squeeze()   # ✅ correct
        print(f"  neurolib runtime: {t_neurolib:.3f} s")
    else:
        print(f"  ❌ outputs is not a dict – cannot extract 'E' and 'I'.")
        if hasattr(neurolib_model.outputs, 'shape'):
            print(f"  outputs shape: {neurolib_model.outputs.shape}")
        print("  Integration did not run correctly. Check the error above.")
        return

    # ==================================================================
    # 4. Your JAX/Diffrax model
    # ==================================================================
    print("\n--- Running JAX/Diffrax model ---")
    # Create model with zero coupling / delays
    jax_model = WilsonCowan(
        state=initial_state,
        dt=dt,
        tau_e=tau_e,
        tau_i=tau_i,
        w_ee=w_ee,
        w_ei=w_ei,
        w_ie=w_ie,
        w_ii=w_ii,
        a_e=a_e,
        a_i=a_i,
        theta_e=theta_e,
        theta_i=theta_i,
        sigma_ou=0.0,
        tau_ou=0.0,
        mean_exc_ou=0.0,
        mean_inh_ou=0.0,
        exc_ext_baseline=baseline_e,
        inh_ext_baseline=baseline_i,    
        fiber_count_matrix=fiber_count_matrix,
        fiber_length_matrix=fiber_length_matrix,
        seed=seed,
    )

    def deterministic_simulate(model, duration):
        t0, t1 = 0.0, duration
        ts = jnp.arange(t0, t1, model.dt)
        # Create a dummy history: shape (num_unique_delays, 2, N) – all zeros.
        # Since connectivity is zero, its content is never actually used.
        dummy_history = jnp.zeros(
            (model.unique_delays.shape[0], 2, model.number_of_regions)
        )
        def dynamics_wrapper(t, y, args):
            # y: (2, N)  – only E and I states
            return model.dynamics(t, y, args, history=dummy_history)

        term = diffrax.ODETerm(dynamics_wrapper)
        solver = diffrax.Euler()
        stepsize_controller = diffrax.ConstantStepSize()
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=model.dt,
            y0=model.state[:2], # only E and I populations
            args=None,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=stepsize_controller,
            delays=None,
            max_steps=16**4,
        )
        return sol.ts, jnp.array(sol.ys)

    # Warm‑up compilation
    print("  Compiling (first run may take a few seconds)...")
    t0_jax = time.perf_counter()
    _, _ = deterministic_simulate(jax_model, 0.1)  # short warm‑up
    jax.block_until_ready(_)
    t_compile = time.perf_counter() - t0_jax
    print(f"  Compilation time: {t_compile:.3f} s")

    # Timed simulation
    t0_jax = time.perf_counter()
    jax_ts, jax_states = deterministic_simulate(jax_model, duration)
    jax.block_until_ready(jax_states)
    t_jax = time.perf_counter() - t0_jax

    # Extract E and I traces
    jax_E = jax_states[:, 0, 0]
    jax_I = jax_states[:, 1, 0]

    print(f"  JAX/Diffrax runtime (after compilation): {t_jax:.3f} s")
    print(f"  Speedup (neurolib / JAX): {t_neurolib/t_jax:.2f}x")

    # ==================================================================
    # 5. Compare results
    # ==================================================================
    print("\n--- Comparison ---")
    # Interpolate neurolib onto the JAX time grid (should be identical, but safe)
    neurolib_E_interp = interp1d(neurolib_t, neurolib_E, kind='linear',
                                 fill_value='extrapolate')(jax_ts)
    neurolib_I_interp = interp1d(neurolib_t, neurolib_I, kind='linear',
                                 fill_value='extrapolate')(jax_ts)

    # Compute error metrics
    max_diff_E = jnp.max(jnp.abs(jax_E - neurolib_E_interp))
    max_diff_I = jnp.max(jnp.abs(jax_I - neurolib_I_interp))
    mae_E = jnp.mean(jnp.abs(jax_E - neurolib_E_interp))
    mae_I = jnp.mean(jnp.abs(jax_I - neurolib_I_interp))

    print(f"Excitatory population (E):")
    print(f"  Max absolute difference : {max_diff_E:.3e}")
    print(f"  Mean absolute error     : {mae_E:.3e}")
    print(f"Inhibitory population (I):")
    print(f"  Max absolute difference : {max_diff_I:.3e}")
    print(f"  Mean absolute error     : {mae_I:.3e}")

    # Tolerance check
    rtol = 1e-3
    atol = 1e-4
    e_close = jnp.allclose(jax_E[1:], neurolib_E_interp[1:], rtol=rtol, atol=atol)
    i_close = jnp.allclose(jax_I, neurolib_I_interp, rtol=rtol, atol=atol)
    
    print(f"\nWithin tolerance (rtol={rtol}, atol={atol})?")
    print(f"  Excitatory: {'✓ PASS' if e_close else '✗ FAIL'}")
    print(f"  Inhibitory: {'✓ PASS' if i_close else '✗ FAIL'}")

    if e_close and i_close:
        print("\n✅ VALIDATION PASSED: JAX model matches neurolib.")
    else:
        print("\n❌ VALIDATION FAILED: Differences exceed tolerance.")

    # Optional: save plot for visual inspection
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        axes[0].plot(jax_ts, jax_E, 'b-', label='JAX E', linewidth=1)
        axes[0].plot(neurolib_t, neurolib_E, 'r--', label='neurolib E', linewidth=1, alpha=0.7)
        axes[0].set_ylabel('E activity')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(jax_ts, jax_I, 'g-', label='JAX I', linewidth=1)
        axes[1].plot(neurolib_t, neurolib_I, 'm--', label='neurolib I', linewidth=1, alpha=0.7)
        axes[1].set_ylabel('I activity')
        axes[1].set_xlabel('Time (s)')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.suptitle('Wilson‑Cowan: JAX/Diffrax vs neurolib (deterministic, single node)')
        plt.tight_layout()
        plt.savefig('wc_validation_deterministic.png', dpi=150)
        print("\nPlot saved as 'wc_validation_deterministic.png'")
    except ImportError:
        print("\nmatplotlib not installed – skipping plot generation.")

    print("="*70 + "\n")


if __name__ == "__main__":
    test_no_delay_no_noise_vs_neurolib()