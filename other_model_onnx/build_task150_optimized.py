from pathlib import Path
import shutil

import onnx


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "repairs" / "task150.onnx"
DESTINATION = ROOT / "other_model_onnx" / "task150.onnx"


def main() -> None:
    """Preserve the strongest portable exact task150 graph found in the audit.

    A three-node ReduceL2/Cast/ReverseSequence rewrite would have cost 8, but
    the competition's ONNX Runtime CPU kernel rejects spatial time axes. The
    source graph's int32 Range plus Gather is therefore the measured portable
    floor (cost 135); copying it also guarantees no failed experiment remains.
    """
    model = onnx.load(SOURCE)
    onnx.checker.check_model(model, full_check=True)
    shutil.copyfile(SOURCE, DESTINATION)
    print(DESTINATION)


if __name__ == "__main__":
    main()
