# Northstar Algorithm Performance Testing

This branch focuses on performance testing the Northstar algorithm after its recent Python rewrite. In particular, we want to evaluate how the algorithm’s performance scales with GPU type and number when parallelized on AWS.

## Overview

- Step-1: Profile the algorithm using Snakeviz (uses Cprofile and Pstats)
- Step-2: Fix the space and time complexity issues 
