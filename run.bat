if "%1%" == "train" (
    set CUDA_VISIBLE_DEVICES=0 & python run.py train
) else if "%1%" == "eval" (
    set CUDA_VISIBLE_DEVICES=0 & python run.py eval
) else if "%1%"=="test" (
    set CUDA_VISIBLE_DEVICES=0 & python run.py test
) else if "%1%"=="train_debug" (
    python run.py train
) else (
    echo Invalid Option Selected
)