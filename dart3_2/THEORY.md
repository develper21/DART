# DART-3.2 Theory

DART-3.1 established a clean evaluation protocol and showed that short task programs can improve frozen-primitive transfer, but program necessity remained weak.

DART-3.2 therefore asks whether the task program is causally meaningful.

For each trained program step:

1. keep the exact trained model state,
2. replace only that program step by identity,
3. measure the DART behavioral delta,
4. perform an analogous teacher intervention,
5. score how closely the two counterfactual effects agree.

The target program is selected on adaptation/validation data only. Final target test is untouched.
