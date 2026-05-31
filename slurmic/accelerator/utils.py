import subprocess


def nvidia_smi_gpu_memory_stats() -> dict:
    """
    Parse the nvidia-smi output and extract the memory used stats.
    """
    out_dict = {}
    try:
        sp = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        )
        out_str = sp.communicate()
        out_list = out_str[0].decode("utf-8").split("\n")
        out_dict = {}
        for item in out_list:
            if " MiB" in item:
                gpu_idx, mem_used = item.split(",")
                gpu_key = f"gpu_{gpu_idx}_mem_used_gb"
                out_dict[gpu_key] = int(mem_used.strip().split(" ")[0]) / 1024
    except FileNotFoundError:
        raise Exception("Failed to find the 'nvidia-smi' executable for printing GPU stats")
    except subprocess.CalledProcessError as e:
        raise Exception(f"nvidia-smi returned non zero error code: {e.returncode}")

    return out_dict


def nvidia_smi_gpu_memory_stats_str() -> str:
    """
    Parse the nvidia-smi output and return a human-readable GPU memory report.
    """
    stats = nvidia_smi_gpu_memory_stats()
    if not stats:
        return "GPU memory usage: no GPUs detected"

    # Keys look like "gpu_<idx>_mem_used_gb"; recover the index for a tidy label.
    rows = [(key.split("_")[1], value) for key, value in stats.items()]
    width = max(len(idx) for idx, _ in rows)
    lines = [f"  GPU {idx:>{width}}: {value:7.2f} GB" for idx, value in rows]
    return "GPU memory usage:\n" + "\n".join(lines)
