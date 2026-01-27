@echo off
:: Wrapper to run torchrun with USE_LIBUV=0 set for this process
:: Usage: run_torchrun.bat --nproc_per_node=1 main_swav.py --arch resnet18 ...
if "%USE_LIBUV%"=="" (
  set USE_LIBUV=0
)
:: Forward all arguments to torchrun
torchrun %*
