# V19 PCGrad stability veto

This GPU notebook is a suppression-only continuation of the accepted V15_B
submission. It runs the four already-audited V14 PCGrad repairs over the exact
V15_B box bank and exports predeclared agreement variants. It never creates a
Kaggle competition submission.

Hard invariants:

- exact V15_B SHA256 `4345387b72aecd55dd3856b1607ed836ec9f400365fcf3f3b89f8947f1ff2412`;
- 2,000 unique image IDs and exactly 3,995 boxes;
- zero added or moved boxes;
- zero confidence increases;
- frozen thresholds before test enumeration;
- no test-derived training, fitting, selection, labels, or pseudo-labels.
