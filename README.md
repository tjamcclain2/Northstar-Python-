# Subproject-1:

## Northstar Algorithm Performance Testing

Performance testing: The Northstar algorithm has not be tested extensively since its Python rewrite. For example, it is not yet clear how the algorithm’s performance scales with GPU type or number when parallelization is left to AWS. Working on this sub-project would require mastering the AWS console and learning to start, run, and transfer data to and from AWS, which constitute the nuts and bolts of high-performance computing in the cloud.

## Overview

- Step-1: Profile the algorithm using Snakeviz (uses Cprofile and Pstats) [done]
- Step-2: Optimize the algorithm at code level --> Goal: Make sure you have algorithmic efficiency
- Step-3: Optimize at server-level --> Is my algorithm giving the best performance across all GPU instances?
