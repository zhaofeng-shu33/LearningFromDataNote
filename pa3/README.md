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
# Grading details
The grading consists of three part:
* auto-grade (8.5) by running `python test.py`
* analysis for kmeans (2)
    * right figure (1 points)
    * plausible analysis (1 points)
* analysis for spectral clustering (2)
    * right figure and meaningful gamma (1 points)
    * plausible analysis for why gamma cannot be too small (0.5 points)
    * plausible analysis for why gamma cannot be too large (0.5 points)
## Common pitfalls to lose points for analysis problem
* forget to upload the figure.
* Mainly use random initialization of kmeans to explain the cause

# Analysis problem
## Kmeans
Some students think random initialization is the main problem to achieve unexpected result.
I think this explanation is not hitting the essence of problem. Even using
`kmeans++` as initialization it is still hard to get a good result. That is to say that with
large probability we could not get the expected result. Therefore, the main problem lies with the
formulation of kmeans itself and not the randomness of the algorithm. You can regard the randomness
as a minor factor by saying that choosing initial centroid near [1, 0], [5, 0] can produce the
human-expected result.
## Spectral Clustering
Some students think the gamma between 1100 ~ 2000 can not always produce expected results.
This is due to random initialization of kmeans, not the problem of spectral clustering algorithm itself. If you use sklearn (normalized) implementation, you won't have the random result.