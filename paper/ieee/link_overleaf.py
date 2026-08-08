#!/usr/bin/env python3
"""Point this folder at an Overleaf project, so the Overleaf Workshop
extension treats it as a Local Replica.

The extension recognises a folder as a replica when it contains
.overleaf/settings.json holding a URI of the form

    overleaf-workshop://<server>/<ProjectName>?user%3D<userId>%26project%3D<projectId>

The user id is read from VS Code's own extension state, so nothing has to
be typed twice and no credential is copied into this folder.

This folder is reached two ways -- directly, and through the symlink at
~/Downloads/OverleafProjects/BipedSim2Sim that follows the extension's
convention. Path resolution follows symlinks, so the directory name is not
a reliable project name and is passed explicitly instead.

Usage, after creating the project on Overleaf:

    python3 link_overleaf.py https://www.overleaf.com/project/<id>
    python3 link_overleaf.py <id> "My Project Name"
"""

import json
import pathlib
import re
import sqlite3
import sys
from urllib.parse import quote

HERE = pathlib.Path(__file__).resolve().parent
STATE = pathlib.Path.home() / ".config/Code/User/globalStorage/state.vscdb"
SERVER = "www.overleaf.com"


def project_id(arg: str) -> str:
    """Accept a full project URL or a bare 24-hex id."""
    m = re.search(r"[0-9a-f]{24}", arg)
    if not m:
        raise SystemExit(
            f"could not find a 24-character project id in {arg!r}.\n"
            "Open the project on Overleaf and copy the URL; it looks like\n"
            "  https://www.overleaf.com/project/0123456789abcdef01234567")
    return m.group(0)


def user_id() -> str:
    if not STATE.exists():
        raise SystemExit(f"VS Code state not found at {STATE}")
    con = sqlite3.connect(f"file:{STATE}?mode=ro", uri=True)
    row = con.execute(
        "SELECT value FROM ItemTable WHERE key='iamhyc.overleaf-workshop'").fetchone()
    if not row:
        raise SystemExit("Overleaf Workshop has no stored login. "
                         "Log in from the extension panel first.")
    v = row[0]
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    d = json.loads(v)
    try:
        return d["overleaf-servers"][SERVER]["login"]["userId"]
    except KeyError:
        raise SystemExit(f"no login found for {SERVER} in the extension state.")


DEFAULT_NAME = "BipedSim2Sim"


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    pid = project_id(sys.argv[1])
    uid = user_id()
    # Never HERE.name: this file is reached through a symlink whose target
    # directory is called "ieee", which is not the Overleaf project name.
    name = sys.argv[2] if len(sys.argv) == 3 else DEFAULT_NAME

    # The query string is doubly encoded: the extension stores '=' as %3D and
    # '&' as %26 inside the URI, matching how it writes its own replicas.
    query = f"user{quote('=')}{uid}{quote('&')}project{quote('=')}{pid}"
    uri = f"overleaf-workshop://{SERVER}/{name}?{query}"

    out = HERE / ".overleaf"
    out.mkdir(exist_ok=True)
    settings = {
        "uri": uri,
        "serverName": SERVER,
        "enableCompileNPreview": False,
        "projectName": name,
    }
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    print(f"linked {HERE} -> Overleaf project {pid[:6]}...{pid[-4:]}")
    print("\nNow, in VS Code:")
    print(f"  1. File > Open Folder > {HERE}")
    print("  2. The Overleaf Workshop panel will pick it up as a Local Replica.")
    print("  3. Use the panel's sync action to push these files up.")


if __name__ == "__main__":
    main()
