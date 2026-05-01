# Try 1

These numbers are crazy. IDK where these sites are getting info from but it doesn't agree with official nvidia spec sheets
 
## 2026

experiment ran on L4: https://gpucost.org/gpu/l4#cloud-pricing, released 2023

20 TFlops/$$ - `20 * 10**12`

experiment took about 24 hours of gpu compute, at 60 TFlops

`24*60*60 seconds * (60 * 10**12) -> 5e18 Flop`


## 2016

https://aiimpacts.org/current-flops-prices/

about 10 GFlops/$ - `10 * 10**9`

Assume reasonable server gpu for comparison, such as Nvidia P100: https://ourworldindata.org/grapher/gpu-price-performance

1GFlops - 1e9


## Difference over ten years

### Cost:

`2e13 / 1e10 = 2e3`

$100 2026 cost -> $200,000 2016 cost

### Time:

`5e18 Flop / 1e9 Flops -> 5e9 seconds -> 160 years`

# Try 2

## 2026 - L4

https://resources.nvidia.com/en-us-data-center-overview/l4-gpu-datasheet

L4 new is about $1/hr on aws, cost new about $7000

https://instances.vantage.sh/aws/ec2/g6.xlarge?currency=USD

So I actually used a g6.8xlarge for the extra cpus but gpu compute is what is important for NN training so using the cheapest quote for the GPU for most conservative estimate

https://gpupoet.com/gpu/learn/card/nvidia-l4

L4 performance

FP32 30E12 Flops
TF32 120E12 Flops  

assuming partial TF32 usage, call it 90E12 Flops

24 hours of compute, about $24
total Flops - `90E12*60*60*24 -> 7.7e18 Flops`

# 2016 - P100

https://images.nvidia.com/content/tesla/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf

https://www.microway.com/hpc-tech-tips/nvidia-tesla-p100-price-analysis/

Price new also about $7000! Cute

assume roughly same cloud price to new price ratio, so $1/hr?

https://www.servethehome.com/google-cloud-platform-now-nvidia-tesla-p100-gpu-nodes/ says $2/hr more normal for on-demand

P100 FP32 4.7E12Flops

Total compute time `7.7E18 / 4.7E12 -> 1.6E6 seconds -> 450 hours`

$450 hours -> $900

that seems more reasonable
