import numpy as np
import pandas as pd
from pathlib import Path


def generate_rsrp_sequence(
    length=1000,
    initial_rsrp=-70.0,
    degradation_rate=0.4,
    noise_std=1.5,
):
    t = np.arange(length)

    noise = np.random.normal(
        0.0,
        noise_std,
        size=length,
    )

    rsrp = (
        initial_rsrp
        - degradation_rate * t
        + noise
    )

    return rsrp


def main():
    np.random.seed(42)
    rsrp = generate_rsrp_sequence()

    df = pd.DataFrame({
        "time": np.arange(len(rsrp)),
        "rsrp": rsrp,
    })

    output = Path("data/raw/rsrp_sample.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(df.head())
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
