# Wrapper to run torchrun with USE_LIBUV=0 set for this process
# Usage: .\run_torchrun.ps1 --nproc_per_node=1 main_swav.py --arch resnet18 ...
param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ForwardArgs
)

# Only set if not already set in the environment
if (-not $env:USE_LIBUV) {
    $env:USE_LIBUV = '0'
}

# Invoke torchrun with forwarded args
& torchrun @ForwardArgs
