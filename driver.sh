#!/bin/bash
dvc pull
dvc dag
dvc repro
dvc plots show -o results
cd results && python -m http.server 8000


# Still working on this. 
# dvc exp run -S params.n_estimators=10,30,50,100 -S params.max_depth=1,3,5,10,50,100 --queue
# dvc exp run --run-all --jobs 1
# dvc exp show --no-pager --sort_by=metrics.accuracy --sort_order=desc
# echo("Best accuracy: $(dvc exp show --no-pager --sort_by=metrics.accuracy --sort_order=desc | tail -n 1 | awk '{print $5}')")
# dvc exp apply best
