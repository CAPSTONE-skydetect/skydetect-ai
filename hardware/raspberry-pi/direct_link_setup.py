"""Root-only Pi maintenance: backed-up link-local setup with timed rollback.

Run through direct_link_setup.sh on the Mac. No sudoers/firewall changes.
The verification checks network reachability, not camera identity or latency.
"""
import argparse
import configparser
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import time

UUID = "e1a61b17-2055-31e2-94ff-287a1193e555"
MAC_PEER = "fe80::1090:eccd:57ad:dd3f%eth0"
BASE = Path("/var/lib/skydetect/network-backups")


def run(*args, check=True):
    return subprocess.run(args, text=True, capture_output=True, check=check, timeout=45)


def value(field):
    return run("nmcli", "-g", field, "connection", "show", UUID).stdout.strip()


def record(folder, **values):
    path = folder / "result.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(values)
    temporary = folder / "result.tmp"
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps({"backup": str(folder), **values}), flush=True)


def restore(folder):
    original = json.loads((folder / "original.json").read_text())
    run("nmcli", "connection", "modify", UUID,
        "ipv4.method", original["ipv4.method"],
        "ipv6.method", original["ipv6.method"])
    run("systemctl", "stop", original["rollback_unit"] + ".timer")
    result = run("nmcli", "--wait", "30", "connection", "up", UUID, "ifname", "eth0", check=False)
    record(folder, status="rolled_back", activation_exit_code=result.returncode)


def probe():
    state = run("nmcli", "-g", "GENERAL.STATE", "device", "show", "eth0").stdout.strip()
    active = run("nmcli", "-g", "GENERAL.CON-UUID", "device", "show", "eth0").stdout.strip()
    reply = run("ping", "-6", "-c", "1", "-W", "2", MAC_PEER, check=False)
    details = {"state": state, "active_uuid": active, "ping_exit": reply.returncode,
               "ping_output": (reply.stdout + reply.stderr)[-1200:]}
    return state.startswith("100") and active == UUID and reply.returncode == 0, details


def retire_completed_rollbacks():
    for path in BASE.glob("direct-*/original.json"):
        original = json.loads(path.read_text())
        unit = original["rollback_unit"]
        if not re.fullmatch(r"skydetect-direct-[0-9]+-rollback", unit):
            raise RuntimeError("Unexpected rollback unit")
        result_file = path.with_name("result.json")
        result = json.loads(result_file.read_text()) if result_file.exists() else {}
        if result.get("status") not in ("rolled_back", "network_verified"):
            raise RuntimeError("An earlier maintenance run is unresolved: " + str(path.parent))
        for service in (unit, unit.removesuffix("-rollback") + "-verify"):
            if run("systemctl", "is-active", service + ".service", check=False).returncode == 0:
                raise RuntimeError("Earlier maintenance service is still active")
        run("systemctl", "stop", unit + ".timer")


def verify(folder):
    original = json.loads((folder / "original.json").read_text())
    try:
        run("nmcli", "--wait", "30", "connection", "up", UUID, "ifname", "eth0")
        # IPv6 DAD/neighbor resolution may lag NM's activated state. This grace
        # period is not counted toward the subsequent uninterrupted check.
        for attempt in range(21):
            ok, details = probe()
            record(folder, status="warming_up", probe=details)
            if ok:
                break
            if attempt == 20:
                raise RuntimeError("Direct link did not become reachable during warm-up")
            time.sleep(3)
        # Four minutes exceeds the previous 45-second DHCP activation cycle.
        for sample in range(25):
            ok, details = probe()
            record(folder, status="verifying", probe=details)
            if not ok:
                raise RuntimeError("Direct link verification failed")
            record(folder, status="verifying", passed_samples=sample + 1)
            if sample < 24:
                time.sleep(10)
        run("systemctl", "stop", original["rollback_unit"] + ".timer")
        record(folder, status="network_verified", duration_seconds=240,
               camera_validation="pending on Mac")
    except Exception as exc:
        record(folder, error=str(exc))
        restore(folder)
        raise


def apply(host_epoch, remote_start):
    if value("connection.type") != "802-3-ethernet":
        raise RuntimeError("Expected Ethernet profile")
    if value("ipv4.method") != "auto" or value("ipv6.method") != "auto":
        raise RuntimeError("Profile is no longer the inspected auto/auto baseline; inspect first")
    retire_completed_rollbacks()
    # Preserve the profile UUID and address-generation settings so IPv6 stays stable.
    folder = BASE / ("direct-" + str(time.time_ns()))
    folder.mkdir(parents=True, mode=0o700)
    folder.chmod(0o700)
    found = False
    for parent in (Path("/etc/NetworkManager/system-connections"), Path("/run/NetworkManager/system-connections")):
        for path in parent.glob("*"):
            if not path.is_file():
                continue
            parser = configparser.ConfigParser(interpolation=None, strict=False)
            parser.read(path)
            if parser.get("connection", "uuid", fallback="") == UUID:
                shutil.copy2(path, folder / ("original-" + str(int(found)) + ".nmconnection"))
                found = True
    if not found:
        raise RuntimeError("No persisted keyfile found; refusing an unbacked change")
    unit = "skydetect-direct-" + str(time.time_ns())
    original = {field: value(field) for field in ("connection.id", "ipv4.method", "ipv6.method")}
    original["rollback_unit"] = unit + "-rollback"
    (folder / "original.json").write_text(json.dumps(original, indent=2) + "\n")
    (folder / "profile-before.txt").write_text(run("nmcli", "connection", "show", UUID).stdout)
    script = folder / "maintenance.py"
    shutil.copyfile(__file__, script)
    script.chmod(0o700)
    # Timer runs independently of SSH if activation interrupts the session.
    run("systemd-run", "--unit=" + original["rollback_unit"], "--on-active=10m",
        "/usr/bin/python3", str(script), "rollback", str(folder))
    try:
        ntp = run("timedatectl", "show", "-p", "NTP", "--value").stdout.strip()
        corrected = host_epoch + time.clock_gettime(time.CLOCK_BOOTTIME) - remote_start
        run("timedatectl", "set-ntp", "false")
        try:
            run("date", "--set=@" + str(corrected))
        finally:
            if ntp == "yes":
                run("timedatectl", "set-ntp", "true")
        record(folder, status="prepared", time_source="Mac wall clock; includes sudo wait compensation")
        run("nmcli", "connection", "modify", UUID, "ipv4.method", "link-local", "ipv6.method", "link-local")
        run("systemd-run", "--unit=" + unit + "-verify",
            "/usr/bin/python3", str(script), "verify", str(folder))
    except Exception:
        restore(folder)
        raise
    print("Backup and status: " + str(folder), flush=True)
    print("Network-only verification is running for 4 minutes; failure restores the old methods.", flush=True)
    print("Manual rollback: sudo python3 " + str(script) + " rollback " + str(folder), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["apply", "verify", "rollback"])
    parser.add_argument("argument")
    parser.add_argument("remote_start", nargs="?", type=float)
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise SystemExit("Run with sudo on the Pi; do not share the password in chat.")
    if args.action == "apply":
        if args.remote_start is None:
            parser.error("apply requires the remote boot-time start")
        apply(float(args.argument), args.remote_start)
    else:
        folder = Path(args.argument).resolve()
        if folder.parent != BASE or not folder.is_dir():
            raise SystemExit("Expected an existing root-owned backup directory")
        (verify if args.action == "verify" else restore)(folder)
