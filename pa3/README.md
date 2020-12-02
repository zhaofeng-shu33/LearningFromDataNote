# FAQ
Below are common questions TA received from students when doing PA3.

* When it shows a green check mark on your GitHub project root page, it shows that you can get the partial score (6 points or 8.5 points depending on whether you finish the bonus question). The other 4 points (analysis problem) will be graded manually depending on your modification of `README.md` (or equivalent upload).

* For the first problem, be careful with the initialization of `cluster_centers_` and make sure its type is `float` instead of `int`.
Otherwise, strange assignment will happen and you may not pass the first
test.

* For the second problem, you should implement `_get_affinity_matrix` by using Numpy API instead of writing double loop. Otherwise, you could not pass the speed test.

* Do not modify `test.py` in your submission.

* Do not ask TA about specific implementation of the programming assignments.

* If the X mark shows up on your GitHub project root page, please click the X mark to see the error detail.

* If due to some unknown reason, the auto-grading is not enabled in the repository of some students. If this happens, please click `Actions -- Enable GitHub Actions` by yourself.
