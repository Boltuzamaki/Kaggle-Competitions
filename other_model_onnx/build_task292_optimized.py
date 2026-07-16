from pathlib import Path
import shutil

import onnx


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "repairs" / "task292.onnx"
DESTINATION = ROOT / "other_model_onnx" / "task292.onnx"


def main() -> None:
    """Preserve the audited dense-parameter floor for task292.

    The graph is already one exempt-output Einsum with no charged activation
    memory. Its two independent channel modes and two independent spatial
    modes are both necessary, leaving 20 + 60 dense coefficients. Factoring
    either matrix further increases the total parameter count.
    """
    model = onnx.load(SOURCE)
    onnx.checker.check_model(model, full_check=True)
    shutil.copyfile(SOURCE, DESTINATION)
    print(DESTINATION)


if __name__ == "__main__":
    main()
