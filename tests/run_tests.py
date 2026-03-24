"""
Standalone test runner for the Snake environment.

Works without pytest — uses plain asserts and introspection.
Run with: python tests/run_tests.py

For your local machine, you can also use:
    pytest tests/test_env.py -v
"""

import sys
import os
import traceback
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_env import *


def run_all_tests():
    """Discover and run all test classes and methods."""
    # Find all test classes in the test module
    import test_env
    test_classes = [
        obj for name, obj in vars(test_env).items()
        if isinstance(obj, type) and name.startswith("Test")
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    print("=" * 70)
    print("SNAKE ENVIRONMENT TEST SUITE")
    print("=" * 70)

    start_time = time.time()

    for cls in test_classes:
        print(f"\n--- {cls.__name__} ---")

        # Get all test methods
        methods = [m for m in dir(cls) if m.startswith("test_")]

        for method_name in methods:
            total += 1
            # Create instance and set up fixtures
            instance = cls()

            # Check if method expects fixtures
            method = getattr(instance, method_name)
            import inspect
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())

            try:
                kwargs = {}
                if "env" in params:
                    kwargs["env"] = SnakeEnv(grid_size=20, seed=42)
                if "small_env" in params:
                    kwargs["small_env"] = SnakeEnv(grid_size=6, seed=42)

                method(**kwargs)
                passed += 1
                print(f"  ✓ {method_name}")
            except AssertionError as e:
                failed += 1
                errors.append((cls.__name__, method_name, e, traceback.format_exc()))
                print(f"  ✗ {method_name} — ASSERTION FAILED")
            except Exception as e:
                failed += 1
                errors.append((cls.__name__, method_name, e, traceback.format_exc()))
                print(f"  ✗ {method_name} — ERROR: {e}")

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed, {total} total ({elapsed:.2f}s)")
    print("=" * 70)

    if errors:
        print("\n--- FAILURES ---\n")
        for cls_name, method_name, err, tb in errors:
            print(f"{cls_name}.{method_name}:")
            print(f"  {err}")
            print()

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
