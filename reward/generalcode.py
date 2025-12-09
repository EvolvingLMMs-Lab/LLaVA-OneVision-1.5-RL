import json
import os
import re
import subprocess
import tempfile
import textwrap
from typing import Tuple

from .utils import extract_boxed_content


def run_untrusted_code(code: str, timeout: float = 2.0,
                       mem_mb: int = 50, cpu_sec: int = 2) -> Tuple[int, str, str]:
    """
    Run `code` in a separate Python process with basic Linux resource limits.
    Returns (returncode, stdout, stderr).
    """
    # basic static scan to reject filesystem / move / remove / subprocess usage in untrusted code
    lc = code.lower()
    forbidden_signatures = [
        "os.remove", "os.unlink", "os.rmdir", "os.rename", "os.replace",
        "shutil.", "pathlib", "path.", "open(", "os.system", "subprocess",
        "fopen(", "rm -", "rmtree", "shutil.move", "shutil.copy", ".unlink(", ".rmdir(", ".rename("
    ]
    detected = [sig for sig in forbidden_signatures if sig in lc]
    if detected:
        return -2, "", "Rejected: filesystem or subprocess-related operations detected: " + ", ".join(sorted(set(detected)))

    # write the untrusted code to a temporary file in /dev/shm for faster in-memory IO
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, dir='/dev/shm', prefix='untrusted_') as tf:
        user_code_path = tf.name
        tf.write(code)

    # create a shim that sets resource limits and drops privileges, then executes the file
    shim = textwrap.dedent(f"""
        import resource, os, sys
        # set address space (virtual memory) limit
        mem_bytes = {mem_mb} * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        # set CPU time limit (seconds)
        resource.setrlimit(resource.RLIMIT_CPU, ({cpu_sec}, {cpu_sec}))
        # optionally drop privileges if running as root
        try:
            import pwd
            pw = pwd.getpwnam('nobody')
            os.setgid(pw.pw_gid)
            os.setuid(pw.pw_uid)
        except Exception:
            pass
        # execute the untrusted script
        with open({user_code_path!r}) as f:
            src = f.read()
        # run as __main__
        exec(compile(src, {user_code_path!r}, 'exec'), {{'__name__': '__main__'}})
    """)

    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as shimf:
        shim_path = shimf.name
        shimf.write(shim)

    try:
        # run the shim in a new Python interpreter; -I gives some isolation from environment
        proc = subprocess.run(
            ["python3", "-I", shim_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return -1, e.stdout or "", (e.stderr or "") + "\nTimeoutExpired"
    finally:
        try:
            os.unlink(user_code_path)
            os.unlink(shim_path)
        except OSError:
            pass

def general_code_reward_fn(completions: str, answer: str) -> float:
    boxed_matches = extract_boxed_content(completions)
    if boxed_matches:
        predicted_code = boxed_matches[-1].strip()
        if "```python" in predicted_code:
            # extract content between the first ```python fence and the next ```
            # try to extract the first code block fenced by ```python ... ```
            m = re.search(r"```python(?:\r?\n)?([\s\S]*?)```", predicted_code)
            if m:
                predicted_code = m.group(1).strip()
            else:
                # if there's an opening fence but no closing fence, take everything after the fence
                m2 = re.search(r"```python(?:\r?\n)?([\s\S]*)$", predicted_code)
                if m2:
                    predicted_code = m2.group(1).strip()
                else:
                    # fallback: remove the fence token if present and trim
                    predicted_code = predicted_code.partition("```python")[-1].strip()

        try:
            answer = json.loads(answer)

            for test in answer["tests"]:
                input_data = test["input"]
                expected_output = test["output"]

                exec_code = f"input = \"\"\"{input_data}\"\"\"\n" + predicted_code + "\nprint(solution(input))"

                rc, out, err = run_untrusted_code(exec_code, timeout=2.0)

                if rc != 0:
                    return 0.0
                output = out.strip()
                if output != str(expected_output):
                    return 0.0
            return 1.0
        except Exception:
            return 0.0
    else:
        return 0


# example usage
if __name__ == "__main__":
    rc, out, err = run_untrusted_code("print('hello')\nimport time\ntime.sleep(2)\nprint('done')", timeout=3)
    print("RC:", rc)
    print("OUT:", out)
    print("ERR:", err)