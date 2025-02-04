# Northstar Algorithm Performance Testing

This branch focuses on performance testing the Northstar algorithm after its recent Python rewrite. In particular, we want to evaluate how the algorithm’s performance scales with GPU type and number when parallelized on AWS.

## Overview

The goals of this performance testing project are to:

- **Test Scaling:** Evaluate how performance scales with different GPU instance types and multiple GPUs.
- **Parallel Execution:** Understand the behavior when running the algorithm in parallel across multiple AWS instances.
- **Automation:** Develop automated workflows for launching instances, running benchmarks, and transferring data.
- **HPC in the Cloud:** Gain hands-on experience with AWS services relevant for high-performance computing (HPC), such as EC2, S3, CloudWatch, and AWS Batch/ParallelCluster.

## Roadmap

### 1. Learn the AWS Environment

- **AWS Console & CLI:**
  - Familiarize yourself with launching and managing EC2 instances.
  - Learn how to use the AWS CLI for automated instance management.
- **Training Resources:**
  - Explore the [AWS Getting Started Resource Center](https://aws.amazon.com/getting-started/).
  - Consider AWS training courses (e.g., AWS Solutions Architect or HPC workshops).

### 2. Prepare the GPU-Enabled Environment

- **Select GPU Instance Types:**
  - Evaluate GPU instances like **P2**, **P3**, or **G4** based on your workload.
  - Consider using Spot Instances for cost savings once your workflow is stable.
- **Use Pre-Configured AMIs:**
  - Use AWS Deep Learning AMIs, which come with CUDA, cuDNN, and popular Python libraries pre-installed.
- **Software Setup:**
  - Install your Python dependencies and verify GPU availability (e.g., run `nvidia-smi`).

### 3. Deploy and Run Your Code on AWS

- **Launch a GPU-Enabled Instance:**
  - Log in to the AWS Console and launch an EC2 instance with a suitable GPU instance type.
  - Configure security groups and SSH access.
- **Transfer Code and Data:**
  - Upload your code and datasets to an S3 bucket.
  - Use the AWS CLI (e.g., `aws s3 cp ...`) or SCP/SFTP to transfer files to your instance.
- **Execute and Instrument Your Code:**
  - Add timers (using Python’s `timeit` module or profilers) and logging to key sections of your algorithm.
  - Monitor GPU usage with `nvidia-smi` and set up CloudWatch for deeper performance monitoring.

### 4. Scale Out and Automate Testing

- **Multi-GPU and Multi-Instance Testing:**
  - Test on instances with multiple GPUs (e.g., p3.8xlarge) to fully utilize available resources.
  - Scale horizontally by running the algorithm across multiple instances.
- **Parallelization:**
  - Consider parallel libraries (e.g., [Dask](https://dask.org/) or MPI for Python) if your algorithm supports distributed processing.
- **Automation:**
  - Use AWS Batch or AWS ParallelCluster to manage job scheduling.
  - Automate instance launch, execution, and termination using AWS CLI scripts, CloudFormation, or Terraform.
  - Set up CloudWatch dashboards and alarms to monitor resource utilization and performance metrics.

### 5. Data Transfer and Analysis

- **Transfer Results:**
  - Use AWS S3 to store benchmark logs and performance metrics.
  - Automate data transfers using commands like `aws s3 sync`.
- **Analyze Performance:**
  - Collect metrics such as execution time, GPU utilization, and throughput.
  - Visualize results using Jupyter notebooks, Amazon QuickSight, or other analysis tools.

### 6. Mastering AWS HPC

- **Learning Resources:**
  - Study AWS documentation and tutorials focused on HPC.
  - Read AWS whitepapers and best practices for HPC.
- **Community Support:**
  - Engage with AWS forums and Stack Overflow.
  - Consider AWS support plans if needed.

## Getting Started

1. **Set up AWS Credentials:**
   - Ensure your AWS CLI is configured with the necessary credentials.
2. **Launch a Test Instance:**
   - Start with a single GPU-enabled instance and verify GPU availability using `nvidia-smi`.
3. **Clone This Repository:**
   ```bash
   git checkout performance-testing

