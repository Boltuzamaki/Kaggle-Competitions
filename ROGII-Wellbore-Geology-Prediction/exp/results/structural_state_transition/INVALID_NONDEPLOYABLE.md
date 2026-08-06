The initial pilot result (RMSE 10.257210) is invalid and must not be used.

That run accidentally included the train-only formation columns ANCC, ASTNU,
ASTNL, EGFDU, EGFDL, and BUDA in transition features. Those fields are not
legal hidden-test inputs. The code was corrected to remove the entire marker
feature block before the legal rerun.
