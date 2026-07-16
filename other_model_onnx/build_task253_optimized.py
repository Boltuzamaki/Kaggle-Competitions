from pathlib import Path
import shutil

import onnx


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "repairs" / "task253.onnx"
DESTINATION = ROOT / "other_model_onnx" / "task253.onnx"


def main() -> None:
    """Preserve the lowest-cost portable exact task253 graph found.

    The source is already the strongest audited formulation: periodic spatial
    signatures identify the four triomino orientations, int8 arithmetic keeps
    intermediates compact, and a 4x4 bool one-hot block is padded only in the
    final free output node. All tested fusions cost more than this cost-395
    representation.
    """
    model = onnx.load(SOURCE)
    onnx.checker.check_model(model, full_check=True)
    shutil.copyfile(SOURCE, DESTINATION)
    print(DESTINATION)


if __name__ == "__main__":
    main()
