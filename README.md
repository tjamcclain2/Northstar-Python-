# Progress Updates

## Subproject-1 (Abid):

## Northstar Algorithm Performance Testing

Performance testing: The Northstar algorithm has not be tested extensively since its Python rewrite. For example, it is not yet clear how the algorithm’s performance scales with GPU type or number when parallelization is left to AWS. Working on this sub-project would require mastering the AWS console and learning to start, run, and transfer data to and from AWS, which constitute the nuts and bolts of high-performance computing in the cloud.

## Workflow:

- Step-1: Profile the algorithm using Snakeviz (uses Cprofile and Pstats) [done]
- Step-2: Optimize the algorithm at code level --> Goal: Make sure you have algorithmic efficiency [done; Week-3 of July]
- Step-3: Optimize at server-level --> Is my algorithm giving the best performance across all GPU instances?

### Week-3 (Pivot Point):

### Performance Optimization Summary

- A **drop in total runtime** from ~42 seconds to <0.5 seconds  
- Elimination of over **26 million redundant calls** to `numpy.dot()`  
- All major bottlenecks removed, with **zero calls recorded** in optimized flame graphs

Please check out the comparative analysis for 100 samples provided in the Profile_Code directory!

## On Week-4, 
### I've decided to pivot to learn the fundamentals of training a neural network for our research.
