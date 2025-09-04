import os
import sys
import yaml
import socket
import importlib
from collections import defaultdict
from typing import Any, List, Dict, Optional
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

import asyncio, time, uuid
from mcp.server.fastmcp import FastMCP

from molscore.manager import PRESETS
from acegen.script_helpers import create_save_dir

# Initialize FastMCP server
mcp = FastMCP("acegen")

# Algorithm choices
ALGORITHMS = {
    "acegen_molopt": (
        Path(__file__).parent / "acegen" / "ag.py",
        Path(__file__).parent / "acegen" / "config_molopt_denovo.yaml",
    ),
    "acegen_pract": (
        Path(__file__).parent / "acegen" / "ag.py",
        Path(__file__).parent / "acegen" / "config_practical_denovo.yaml",
    ),
    "a2c": (
        Path(__file__).parent / "a2c" / "a2c.py",
        Path(__file__).parent / "a2c" / "config_denovo.yaml",
    ),
    "ahc": (
        Path(__file__).parent / "ahc" / "ahc.py",
        Path(__file__).parent / "ahc" / "config_denovo.yaml",
    ),
    "augmem": (
        Path(__file__).parent / "augmented_memory" / "augmem.py",
        Path(__file__).parent / "augmented_memory" / "config_denovo.yaml",
    ),
    "dpo": (
        Path(__file__).parent / "dpo" / "dpo.py",
        Path(__file__).parent / "dpo" / "config_denovo.yaml",
    ),
    "hill_climb": (
        Path(__file__).parent / "hill_climb" / "hill_climb.py",
        Path(__file__).parent / "hill_climb" / "config_denovo.yaml",
    ),
    "ppo": (
        Path(__file__).parent / "ppo" / "ppo.py",
        Path(__file__).parent / "ppo" / "config_denovo.yaml",
    ),
    "reinforce": (
        Path(__file__).parent / "reinforce" / "reinforce.py",
        Path(__file__).parent / "reinforce" / "config_denovo.yaml",
    ),
    "reinvent": (
        Path(__file__).parent / "reinvent" / "reinvent.py",
        Path(__file__).parent / "reinvent" / "config_denovo.yaml",
    ),
    "screening": (
        Path(__file__).parent / "screening" / "screening.py",
        Path(__file__).parent / "screening" / "config.yaml",
    ),
}

JOBS = defaultdict(dict)

# ----- Functions -----
def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0
    
def find_available_port(start_port: int = 8501, max_tries: int = 100) -> int:
    port = start_port
    for _ in range(max_tries):
        if not _is_port_in_use(port):
            return port
        port += 1
    raise OSError(f"No available port found in range {start_port}-{port}")
        
async def _launch_streamlit(job_id: str, output_dir: str, port: int):
    JOBS[job_id].update({
        "st_state": "starting",
        "st_pid": None,
        "st_url": f"http://0.0.0.0:{port}",
        "st_proc": None,
        "st_logs": [],
    })
    
    try:
        cmd = [
            "molscore_monitor",
            output_dir,
            "--server.headless=true",
            f"--server.port={port}",
            "--server.address=localhost",
            "--server.runOnSave=false",
        ]
        
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        JOBS[job_id]["st_pid"] = proc.pid
        JOBS[job_id]["st_proc"] = proc
        JOBS[job_id]["st_state"] = "running"
    except Exception as e:
        JOBS[job_id]["st_state"] = "failed"
        JOBS[job_id]["st_logs"].append(f"[error] {e!r}")
    
        
def _run_algorithm_sync(cfg: "OmegaConf", script_path: os.PathLike) -> None:
    """
    Purely synchronous function that does all heavy work:
    imports, file I/O, config creation, and the module's main().
    """
    # Import
    script_dir = Path(script_path).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    module = importlib.import_module(script_path.stem)

    # Run the algorithm's main
    module.main(cfg)

async def _run_acegen(job_id: str, algorithm: str, task: str, total_smiles: int) -> None:
    # Let the loop flush the job_id response first
    await asyncio.sleep(0)

    # Get algorithm
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algorithm '{algorithm}' is not recognized.")
    script_path, cfg_path = ALGORITHMS[algorithm]
    
    # Load config
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)
    
    # Update task
    cfg["molscore_mode"] = "single"
    cfg["molscore_task"] = task
    
    # Update budget
    cfg["total_smiles"] = total_smiles
    
    # Create save_dir
    save_dir = create_save_dir(cfg, script_path)
    cfg["save_dir"] = str(save_dir)

    JOBS[job_id] = {"state": "running", "output_dir": str(save_dir), "logs": []}
    try:
        launch_streamlit(job_id, port=8501)
        # Offload ALL heavy work to a thread
        await asyncio.to_thread(_run_algorithm_sync, cfg, script_path)
        JOBS[job_id]["state"] = "completed"
    except Exception as e:
        JOBS[job_id]["state"] = "failed"
        JOBS[job_id]["error"] = str(e)


# ----- MCP Tools -----
@mcp.tool()
def list_available_algorithms() -> List[str]:
    """List available algorithms."""
    return list(ALGORITHMS.keys())

@mcp.tool()
def list_available_tasks() -> List[str]:
    """List available tasks."""
    # NOTE: For now, let's get GuacaMol tasks
    guacamol_tasks = [f"GuacaMol:{t.stem}" for t in PRESETS["GuacaMol"].glob("*.json")]
    return guacamol_tasks

@mcp.tool()
def read_default_config(algorithm: str) -> Dict[str, Any]:
    """Read the default configuration for a given algorithm."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algorithm '{algorithm}' is not recognized.")
    
    _, cfg_path = ALGORITHMS[algorithm]
    
    with open(cfg_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return config_dict

@mcp.tool()
async def run_acegen(algorithm: str, task: str, total_smiles: int) -> str:
    """Run the specified algorithm with the given model and total_smiles."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algorithm '{algorithm}' is not recognized.")
    if task not in list_available_tasks():
        raise ValueError(f"Task '{task}' is not recognized.")
    
    job_id = uuid.uuid4().hex[:8]
    asyncio.create_task(_run_acegen(job_id, algorithm, task, total_smiles))
    return job_id
    
@mcp.tool()
def get_job_status(job_id: Optional[str] = None) -> Dict[str, Any]:
    """Get the status of a job."""
    
    if job_id is None:
        ser_dict = JOBS.copy()
        for job_id in ser_dict:
            if 'st_proc' in ser_dict[job_id]:
                ser_dict[job_id].pop('st_proc')
        return dict(ser_dict)
    else:
        ser_dict = JOBS.get(job_id, {"state": "not_found"}).copy()
        if 'st_proc' in ser_dict:
            ser_dict.pop('st_proc')
        return ser_dict
    
@mcp.tool()
def launch_streamlit(job_id: str, port: int = 8501) -> str:
    """Start a Streamlit app."""
    if job_id not in JOBS:
        return f"Job ID {job_id} not_found"
    if JOBS[job_id].get("st_proc", False):
        return f"Job ID {job_id} already has a running Streamlit process."
    if not JOBS[job_id].get("output_dir", False):
        return f"Job ID {job_id} has no output directory."
    output_dir = JOBS[job_id]["output_dir"]
    
    if _is_port_in_use(port):
        port = find_available_port(port)
    
    asyncio.create_task(_launch_streamlit(job_id, output_dir, port))
    return get_job_status(job_id)

@mcp.tool()
def stop_streamlit(job_id: str) -> str:
    """Stop a Streamlit app."""
    if job_id not in JOBS:
        return f"Job ID {job_id} not_found"
    if not JOBS[job_id].get("st_proc", False):
        return f"Job ID {job_id} has no running Streamlit process." 
    proc = JOBS[job_id]["st_proc"]
    
    if proc.returncode is None:
        proc.terminate()
        JOBS[job_id]["st_state"] = "stopped"
        JOBS[job_id]["st_proc"] = None
        JOBS[job_id]["st_pid"] = None
        JOBS[job_id]["st_url"] = None
        return "Streamlit terminated"
    else:
        JOBS[job_id]["st_state"] = "not_running"
        return "Streamlit not running"


if __name__ == "__main__":
    mcp.run(transport="stdio")