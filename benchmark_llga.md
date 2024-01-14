# Installation

## Create Conda Environment

```
conda create -n pytorch python=3.10
```

## Common Dependencies

```
conda install -y cmake ninja cython dataclasses future ipython numpy typing typing_extensions Pillow pkg-config pyyaml setuptools openssl
conda install mkl intel-openmp mkl-include -c intel --no-update-deps
```

## Build Pytorch

```
pip install -r requirements.txt

USE_CUDA=0 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 python setup.py develop
```

check:

```
cd ..
python
import torch
```

## Build Torchvision

```
conda install jpeg libpng
export LD_LIBRARY_PATH=/home/jiexinzh/anaconda3/envs/pytorch/lib/libjpeg.so:/home/jiexinzh/anaconda3/envs/pytorch/lib/libpng.so:${LD_LIBRARY_PATH}
 (export LD_LIBRARY_PATH=/path/to/libjpeg:/path/to/libpng:${LD_LIBRARY_PATH})
git clone https://github.com/pytorch/vision.git --branch main --recursive
cd vision
// export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```

check:

```
cd ..
python
import torchvision
```

## Install Jemalloc

```
conda install jemalloc
```

(Only one of jemalloc/tcmalloc is needed)

## Build tcmalloc

```
wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.7.90/gperftools-2.7.90.tar.gz
tar -xzf gperftools-2.7.90.tar.gz
cd gperftools-2.7.90
./configure --prefix=$HOME/.local
make
make install
```

## Build Torchbench

```
conda install -y git-lfs
git clone https://github.com/sanchitintel/benchmark.git --branch onednn_graph_benchmark --recursive
cd benchmark
```

Remember to remove torchtext in `torchbench/benchmark/utils/__init__.py` before building torchbench:
```
#TORCH_DEPS = ['torch', 'torchvision', 'torchaudio']
TORCH_DEPS = ['torch', 'torchvision']
```

Avoid installation interruption when some models setup fail:
```
parser.add_argument("--continue_on_fail", action="store_false")
```

```
python install.py
```

# Get Performance

## Quick Start

To get performance of each model, just run `llga_benchmark_fp32.sh` or `llga_benchmark_bf16.sh` with the following command:

```
For example:
bash llga_benchmark_fp32.sh
```

In the following sessions, we will explain the setting of each environment variable.

To profile or debug single model, please refer to [Single Model Profiling or Debugging](#single-model-profiling-or-debugging).

## Memory Allocator Setting

PyTorch uses dynamic graph which has a flaw that output of each operator must be allocated for each execution, which increases the burden of memory allocation and will trigger clear page for large buffer. For deep learning workloads, Jemalloc or TCMalloc can get better performance by reusing memory as much as possible than default malloc funtion. One of them is holding memory in caches to speed up access of commonly-used objects.

For tcmalloc:

```
export LD_PRELOAD=/home/jiexinzh/.local/lib/libtcmalloc.so:$LD_PRELOAD
```

But for jemalloc, you also need to setup `MALLOC_CONF` first:

```
export MALLOC_CONF="oversize_threshold:1 
background_thread:true,
metadata_thp:auto,
dirty_decay_ms:-1,muzzy_decay_ms:-1"
```

Then:

```
export LD_PRELOAD=/home/jiexinzh/anaconda3/envs/pytorch/lib/libjemalloc.so:$LD_PRELOAD
```

Note that we set both `dirty_decay_ms` and `muzzy_decay_ms` with `-1` here, which means page purging is disabled.
For more details about jemalloc runtime option performance tuning, please refer to [jemalloc_tuning](https://github.com/jemalloc/jemalloc/blob/dev/TUNING.md).

## OpenMP Setting

OpenMP is utilized to bring better performance for parallel computation tasks, we can use the following environment variables to get better performance:

* KMP_AFFINITY
* KMP_BLOCKTIME
* OMP_NUM_THREADS
* KMP_SETTINGS

Generally, we can set these environment variables in this way:

```
KMP_AFFINITY=granularity=fine,verbose,compact,1,0 
KMP_BLOCKTIME=1 
KMP_SETTINGS=1 
OMP_NUM_THREADS=<num_physical_cores>
```

### OMP_NUM_THREADS

This environment variable sets the maximum number of threads to use for OpenMP parallel regions if no other value is specified in the application. You can take advantage of this setting to fully squeeze computation capability of your CPU.

The default value is the number of logical processors visible to the operating system on which the program is executed. This value is recommended to be set with the number of physical cores.

With Hyperthreading enabled, there are more than one hardware threads for a physical CPU core, but we recommend to use only one hardware thread for a physical CPU core to avoid cache miss problems.

### KMP_AFFINITY

Users can bind OpenMP threads to physical processing units. KMP_AFFINITY is used to take advantage of this functionality. It restricts execution of certain threads to a subset of the physical processing units in a multiprocessor computer. For more detailed explanation of this environment variable, please refer to [Intel® C++ developer guide.](https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/thread-affinity-interface.html).

### KMP_BLOCKTIME

This environment variable sets the time, in milliseconds, that a thread should wait, after completing the execution of a parallel region, before sleeping.

### KMP_SETTINGS

This environment variable enables (TRUE) or disables (FALSE) the printing of OpenMP run-time library environment variables during program execution.

## NUMA Control

Running on a NUMA-enabled machine brings with it special considerations. NUMA or non-uniform memory access is a memory layout design used in data center machines meant to take advantage of locality of memory in multi-socket machines with multiple memory controllers and blocks. In most cases, inference runs best when confining both the execution and memory usage to a single NUMA node.

In general cases the following command executes a PyTorch script on cores on the Nth node only, and avoids cross-socket memory access to reduce memory access overhead.

```
numactl --cpunodebind=N --membind=N python <pytorch_script>
```

`--physcpubind` can work as well:

```
# e.g. say each socket has 20 cores, to use the 1st socket:
numactl --physcpubind=0-19 --membind=0
```

## Single Model Profiling or Debugging

Sometimes we want to reproduce particular model performance and do profiling/debugging, in this case, you can use `--only` and `--export-profiler-trace --profiler-trace-name`:
:
```
export LD_PRELOAD=/home/sdp/.local/lib/libtcmalloc.so:$LD_PRELOAD && export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:$LD_PRELOAD && export TORCHINDUCTOR_ONEDNN_GRAPH=1 && export KMP_AFFINITY=granularity=fine,verbose,compact,1,0 && export KMP_BLOCKTIME=1 && export KMP_SETTINGS=1 && export OMP_NUM_THREADS=56 && numactl -C 56-111 -m 1 python -u benchmarks/dynamo/torchbench.py --output=benchmark_logs_jiexin/inductor_torchbench_float32_multi_thread_inference_cpu_performance.csv --performance --inference --float32 -n50 -dcpu --inductor --only nvidia_deeprecommender --export-profiler-trace --profiler-trace-name trace_ww44_with_llga >> nvidia_deeprecommender_ww44_with_llga_with_trace.txt 2>&1 &

```

# References

* <https://www.intel.com/content/www/us/en/developer/articles/technical/how-to-get-better-performance-on-pytorchcaffe2-with-intel-acceleration.html>
* <https://www.intel.com/content/www/us/en/developer/articles/technical/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference.html>
* <https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/thread-affinity-interface.html>
* <https://jemalloc.net/jemalloc.3.html>
