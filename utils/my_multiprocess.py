import os
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from arch.config import DEBUG, TQDM_MIN_INTERVAL

def get_cgroup_cpu_count():
    cgroup_v2_path = "/sys/fs/cgroup/cpu.max"
    cfs_quota_path = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
    cfs_period_path = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
    if os.path.exists(cgroup_v2_path):
        with open(cgroup_v2_path) as f:
            data = f.read().split()
            if data[0] == "max":
                return os.cpu_count()  # No limit set
            return float(data[0]) / float(data[1])
    if os.path.exists(cfs_quota_path):
        with open(cfs_quota_path) as f_quota, open(cfs_period_path) as f_period:
            quota = int(f_quota.read())
            period = int(f_period.read())
            if quota == -1:
                return os.cpu_count() # No limit set
            return quota / period

CPU_COUNT = get_cgroup_cpu_count() or os.cpu_count()
CPU_PER_RANK = max(1, int(CPU_COUNT // int(os.environ.get('WORLD_SIZE', 1))))

def my_batch_convert(
    func, input_root: str | Path, output_root: str | Path,
    input_suffix_or_glob_or_rel_paths: str | list[str], output_suffix: str,
    check_empty: bool=False,
    serial: bool=False,
    retry=2,
):
    input_root = Path(input_root).resolve()
    if isinstance(output_root, str) and output_root.startswith('_'):
        output_root = input_root.parent / f"{input_root.name}{output_root}"
    else:
        output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if isinstance(input_suffix_or_glob_or_rel_paths, str):
        if input_suffix_or_glob_or_rel_paths.startswith('.'):
            input_pattern = f"**/*{input_suffix_or_glob_or_rel_paths}"
        else:
            input_pattern = input_suffix_or_glob_or_rel_paths
        input_paths = sorted(input_root.glob(input_pattern))
        if input_suffix_or_glob_or_rel_paths.startswith('.'):
            # Skip hidden folders/files
            input_paths = [
                path for path in input_paths
                if not any(part.startswith('.') for part in path.relative_to(input_root).parts)
            ]
    else:
        input_paths = input_suffix_or_glob_or_rel_paths

    all_tasks: list[tuple[Path, Path]] = []
    for input_path in input_paths:
        rel_path = input_path.relative_to(input_root)
        output_path = (output_root / rel_path).with_suffix(output_suffix)
        all_tasks.append((input_path, output_path))
    if not all_tasks:
        print(f"[{func.__name__}] Warning: No tasks found.")
        return

    if check_empty:
        def is_done(path: Path):
            return path.exists() and not path.stat().st_size == 0
    else:
        def is_done(path: Path):
            return path.exists()

    cur_tasks = all_tasks
    for attempt in range(retry):
        todo = []
        for task in cur_tasks:
            _, output_path = task
            if is_done(output_path):
                continue
            todo.append(task)
        if todo:
            if attempt == 0:
                num_new_tasks = len(todo)
                print(f"[{func.__name__}] {num_new_tasks}/{len(all_tasks)} new tasks to do.")
        else:
            if attempt == 0:
                print(f"[{func.__name__}] All files already processed before.")
                return
            break

        print(f"[{func.__name__}] Attempt {attempt + 1}/{retry}: {len(todo)}/{num_new_tasks} remaining")

        my_multiprocess(func, todo, serial=serial)

        cur_tasks = todo

    final_remain = [input_path for input_path, output_path in cur_tasks if not is_done(output_path)]
    if final_remain:
        print(f"[{func.__name__}]: {len(final_remain)}/{num_new_tasks} files failed after {retry} attempts: {final_remain[:3]}.")
    else:
        print(f"[{func.__name__}]: Successfully processed {num_new_tasks} new files.")

def my_multiprocess(func, tasks, serial=False):
    if len(tasks) == 0:
        print("No tasks to process.")
        return
    if DEBUG:
        args = tasks[0]
        args_repr = tuple(str(arg) if isinstance(arg, (str, Path)) else type(arg) for arg in args)
        print(f"Debug mode, running first task: {args_repr}")
        return [func(*args)]
    if serial:
        return [func(*task) for task in tqdm(tasks, mininterval=TQDM_MIN_INTERVAL)]
    num_processes = min(len(tasks), CPU_PER_RANK)
    print(f"Processing {len(tasks)} tasks using {num_processes} cores...")
    with Pool(processes=num_processes) as pool:
        return pool.starmap(func, tasks)
