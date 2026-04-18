"""Platform detection and OS-level enforcement primitives for the kernel sandbox.

Detects available sandboxing mechanisms:
- macOS: Seatbelt (sandbox-exec)
- Linux: Landlock (kernel 5.13+, via ctypes syscalls)
- Fallback: none (subprocess-only isolation)
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import sys
import sysconfig

logger = logging.getLogger(__name__)


def _expand(p: str) -> str:
    """Expand ~ and resolve to real path."""
    return os.path.realpath(os.path.expanduser(p))


def is_seatbelt_available() -> bool:
    """Check if macOS sandbox-exec is available."""
    if platform.system() != "Darwin":
        return False
    return shutil.which("sandbox-exec") is not None


def is_landlock_available() -> bool:
    """Check if Linux Landlock enforcement is available.

    Performs a real probe: checks OS, kernel version >= 5.13, then attempts
    a syscall to confirm the kernel supports Landlock.

    Returns True only if Landlock syscalls are confirmed working.
    """
    if platform.system() != "Linux":
        return False
    # Check kernel version >= 5.13
    try:
        release = platform.release()
        major = int(release.split(".")[0])
        minor = int(release.split(".")[1].split("-")[0])
        if (major, minor) < (5, 13):
            return False
    except (ValueError, IndexError):
        return False
    # Probe: try creating a minimal ruleset; if ENOSYS/-1 → unavailable
    try:
        _landlock_probe()
        return True
    except (OSError, NotImplementedError):
        return False


def _landlock_probe() -> None:
    """Probe Landlock availability by attempting syscall. Raises OSError if unavailable."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)

    # struct landlock_ruleset_attr { __u64 handled_access_fs; }
    class _RulesetAttr(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    ALL_FS = sum(1 << i for i in range(14))  # all LANDLOCK_ACCESS_FS_* bits
    attr = _RulesetAttr(handled_access_fs=ALL_FS)
    nr = _landlock_syscall_nr("create_ruleset")
    fd = libc.syscall(nr, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if fd < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"landlock_create_ruleset failed: errno={errno}")
    # Close the fd
    os.close(fd)


def _landlock_syscall_nr(name: str) -> int:
    """Return syscall number for Landlock on the current architecture."""
    arch = platform.machine()
    NR: dict[str, dict[str, int]] = {
        "x86_64":  {"create_ruleset": 444, "add_rule": 445, "restrict_self": 446},
        "aarch64": {"create_ruleset": 444, "add_rule": 445, "restrict_self": 446},
        "arm64":   {"create_ruleset": 444, "add_rule": 445, "restrict_self": 446},
    }
    if arch not in NR:
        raise NotImplementedError(f"Landlock syscall numbers not known for arch: {arch}")
    return NR[arch][name]


# Landlock access rights bitmask constants
LANDLOCK_ACCESS_FS_EXECUTE    = (1 << 0)
LANDLOCK_ACCESS_FS_WRITE_FILE = (1 << 1)
LANDLOCK_ACCESS_FS_READ_FILE  = (1 << 2)
LANDLOCK_ACCESS_FS_READ_DIR   = (1 << 3)
LANDLOCK_ACCESS_FS_REMOVE_DIR = (1 << 4)
LANDLOCK_ACCESS_FS_REMOVE_FILE= (1 << 5)
LANDLOCK_ACCESS_FS_MAKE_CHAR  = (1 << 6)
LANDLOCK_ACCESS_FS_MAKE_DIR   = (1 << 7)
LANDLOCK_ACCESS_FS_MAKE_REG   = (1 << 8)
LANDLOCK_ACCESS_FS_MAKE_SYM   = (1 << 9)
LANDLOCK_ACCESS_FS_MAKE_SOCK  = (1 << 10)
LANDLOCK_ACCESS_FS_MAKE_FIFO  = (1 << 11)
LANDLOCK_ACCESS_FS_MAKE_BLOCK = (1 << 12)
LANDLOCK_ACCESS_FS_MAKE_IPC   = (1 << 13)
LANDLOCK_RULE_PATH_BENEATH    = 1


def apply_landlock_restrictions(project_root: str, temp_dir: str, python_executable: str) -> None:
    """Apply Landlock FS restrictions to the current process.

    Must be called after fork (in subprocess) before executing untrusted code.
    Calls prctl(PR_SET_NO_NEW_PRIVS) first to allow restrict_self without
    CAP_SYS_ADMIN.

    Allows:
    - Read: Python stdlib paths (sys.prefix, sys.exec_prefix, sysconfig paths)
    - Read+Execute: Python executable directory
    - Read+Write: temp_dir
    - Read: project_root (read-only access to project files)
    Denies: everything else

    On non-Linux platforms, raises NotImplementedError.
    """
    if platform.system() != "Linux":
        raise NotImplementedError("apply_landlock_restrictions is Linux-only")

    import ctypes
    import sysconfig as _sysconfig

    libc = ctypes.CDLL(None, use_errno=True)

    # prctl(PR_SET_NO_NEW_PRIVS=38, 1, 0, 0, 0) — required before restrict_self
    PR_SET_NO_NEW_PRIVS = 38
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")

    class _RulesetAttr(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class _PathBeneathAttr(ctypes.Structure):
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    ALL_FS = sum(1 << i for i in range(14))
    attr = _RulesetAttr(handled_access_fs=ALL_FS)
    nr_create = _landlock_syscall_nr("create_ruleset")
    nr_add    = _landlock_syscall_nr("add_rule")
    nr_self   = _landlock_syscall_nr("restrict_self")

    ruleset_fd = libc.syscall(nr_create, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if ruleset_fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")

    READ_EXEC  = LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR | LANDLOCK_ACCESS_FS_EXECUTE
    READ_ONLY  = LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR
    READ_WRITE = (READ_ONLY | LANDLOCK_ACCESS_FS_WRITE_FILE
                  | LANDLOCK_ACCESS_FS_MAKE_REG | LANDLOCK_ACCESS_FS_MAKE_DIR)

    def _add_path_rule(path: str, access: int) -> None:
        path = os.path.realpath(path)
        if not os.path.exists(path):
            return
        target = path if os.path.isdir(path) else os.path.dirname(path)
        fd = os.open(target, os.O_PATH | os.O_CLOEXEC)
        try:
            rule = _PathBeneathAttr(allowed_access=access, parent_fd=fd)
            ret = libc.syscall(nr_add, ruleset_fd, LANDLOCK_RULE_PATH_BENEATH,
                               ctypes.byref(rule), 0)
            if ret < 0:
                raise OSError(ctypes.get_errno(), f"landlock_add_rule failed for {path}")
        finally:
            os.close(fd)

    # Allow Python stdlib read paths
    for p in [sys.prefix, sys.exec_prefix] + list(_sysconfig.get_paths().values()):
        if p and os.path.exists(p):
            _add_path_rule(p, READ_ONLY)

    # Allow Python executable directory read+exec
    _add_path_rule(os.path.dirname(os.path.realpath(python_executable)), READ_EXEC)

    # Allow temp_dir read+write (sandbox scratch space)
    _add_path_rule(temp_dir, READ_WRITE)

    # Allow project_root read-only
    if project_root and os.path.exists(project_root):
        _add_path_rule(project_root, READ_ONLY)

    # Restrict self — after this call, no new FS capabilities can be acquired
    ret = libc.syscall(nr_self, ruleset_fd, 0)
    if ret < 0:
        raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")

    os.close(ruleset_fd)


def get_sandbox_type() -> str:
    """Return the sandbox type available on this platform."""
    if is_seatbelt_available():
        return "seatbelt"
    if is_landlock_available():
        return "landlock"
    return "none"
