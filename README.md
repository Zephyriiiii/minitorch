# minitorch
The full minitorch student suite. 

## Environment

This checkout is configured for the CUDA 13.3 runtime and NVIDIA H20 GPUs on
the current server. Create or update the project environment with:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Verify CUDA before running the GPU assignments:

```bash
python -c "from numba import cuda; print(cuda.is_available(), cuda.get_current_device())"
python -m pytest -m task3_3 -q
python -m pytest -m task3_4 -q
```


To access the autograder: 

* Module 0: https://classroom.github.com/a/qDYKZff9
* Module 1: https://classroom.github.com/a/6TiImUiy
* Module 2: https://classroom.github.com/a/0ZHJeTA0
* Module 3: https://classroom.github.com/a/U5CMJec1
* Module 4: https://classroom.github.com/a/04QA6HZK
* Quizzes: https://classroom.github.com/a/bGcGc12k
