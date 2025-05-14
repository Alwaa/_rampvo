import os
import torch

# Default to server, set the desired cuda device below
RAMPVO_ENV = os.environ.get("RAMPVO_ENV", "server").lower()

########### General ##############
# 1200 => Around 10GB; 0 => No bound!
QUEUE_BUFFER_SIZE = 1200 
# When to start the evaluation NB! Queue should load faster than is used on avg!
QUEUE_ASYNC_MIN_SIZE = 400 
#Currently test with pose
IMU_TESTING = False
#################################


### Location Specific & Overrides ###
if RAMPVO_ENV == "server":
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert cuda_visible is not None and len(cuda_visible), "!On the server you need to set enviroment variable!:\n CUDA_VISIBLE_DEVICES\n"

    LOADING_THREAD_TORCH_INTRA_OP_THREAD_NUM = 2
    TARTAN_PATH_PREFIX = "/data/storage"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    LOADING_THREAD_TORCH_INTRA_OP_THREAD_NUM = 1
    TARTAN_PATH_PREFIX    = ""
#####################################


############ SPESIFICS && EDGE CASE HANDLING ##############
GENERAL_TORCH_INTRA_OP_THREAD_NUM = 4
# Seconds untill just starting eval if not growing
# NB! Only triggers is QUEUE_ASYNC_MIN_SIZE is larger than smallest data set event num
QUEUE_ASYNC_STALL_TIMEOUT = 5.0
QUEUE_ASYNC_SLEEP_BETWEEN_STARTUP_CHECKS = 1.0
###########################################################


torch.set_num_threads(GENERAL_TORCH_INTRA_OP_THREAD_NUM) # OpenMP/MKL/BLAS
torch.set_num_interop_threads(4)# Inter-op pool #NOT CURRENTLY A FACTOR? MAYBE FOR TRAINING?

print(f"\nThreads:\n\tintra-op:{torch.get_num_threads()} | inter-op: {torch.get_num_interop_threads()}")
print("CurrentGPU", os.environ.get("CUDA_VISIBLE_DEVICES"))
print(f"\n NB! For [Evaluation] you should see {LOADING_THREAD_TORCH_INTRA_OP_THREAD_NUM} + {GENERAL_TORCH_INTRA_OP_THREAD_NUM}\
      number of threads with active CPU usage.")

n_visible = torch.cuda.device_count()
if n_visible == 0:
    raise RuntimeError(f"No CUDA devices visible exists?\n Is the enviroment '{RAMPVO_ENV}' ?\n")
else:
    name = torch.cuda.get_device_name(0)
    print(f"\nSuccess! Found {n_visible} visible CUDA device(s).")
    print(f"  • Device 0: {name}\n")