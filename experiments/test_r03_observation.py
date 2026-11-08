
import json
import numpy as np
from pathlib import Path


def check_observation(observation):
    """R-03 observation基础合法性检查"""

    if observation is None:
        return False, "Observation is None"

    try:
        obs = np.asarray(observation, dtype=float)
    except (TypeError, ValueError):
        return False, "Invalid data type"

    if obs.shape != (39,):
        return False, f"Wrong shape: {obs.shape}"

    if not np.all(np.isfinite(obs)):
        return False, "Contains NaN or Inf"

    return True, "Observation valid"


def main():

    folders = list(
        Path("references").glob("B*2026-11-04")
    )

    if not folders:
        print("ERROR: observation folder not found")
        return

    data_file = folders[0] / "observation_samples.json"

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    reset = data["reset"]
    after_step = data["after_one_step"]

    samples = {
        "RESET": reset,
        "AFTER_ONE_STEP": after_step
    }

    print("=" * 50)
    print("R-03 Observation Validation")
    print("=" * 50)

    for name, observation in samples.items():

        passed, reason = check_observation(observation)

        print(f"\n{name}")
        print("Shape:", np.asarray(observation).shape)
        print("Result:", "PASS" if passed else "REJECT")
        print("Reason:", reason)

    reset = np.asarray(reset, dtype=float)
    after_step = np.asarray(after_step, dtype=float)

    print("\n" + "=" * 50)
    print("Key Variable Changes")
    print("=" * 50)

    for i in [5, 6, 7, 8, 37, 38]:

        delta = after_step[i] - reset[i]

        print(
            f"Index {i}: "
            f"before={reset[i]:.10g}, "
            f"after={after_step[i]:.10g}, "
            f"delta={delta:.10g}"
        )

    print("\nStep count:", data["step_count"])
    print("Seed:", data["seed"])
    print("\nValidation completed.")


if __name__ == "__main__":
    main()
