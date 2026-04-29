from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def error_collector() -> dict[str, float]:
    """Collect conservation errors from all module tests."""
    return {}


@pytest.fixture(params=[(1, 4, 8), (2, 8, 64), (2, 16, 256), (2, 16, 1024)])
def shape(request: pytest.FixtureRequest) -> tuple[int, int, int]:
    """Tensor shapes (B, N, d) used for softmax variant tests."""
    return request.param


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Print summary table of all conservation errors after tests complete."""
    if hasattr(pytest, "_module_error_results"):
        results = pytest._module_error_results
        if results:
            print("\n" + "=" * 80)
            print("MODULE CONSERVATION ERROR SUMMARY")
            print("=" * 80)
            print(f"{'Module':40s} {'Error':>15s} {'Status':>20s}")
            print("-" * 80)
            for name in sorted(results.keys()):
                error = results[name]
                if error < 1e-5:
                    status = "✓ PASS (exact)"
                elif error < 1.0:
                    status = "⚠ APPROX (DTD)"
                else:
                    status = "✗ FAIL"
                print(f"{name:40s} {error:>15f} {status:>20s}")
            print("=" * 80 + "\n")
