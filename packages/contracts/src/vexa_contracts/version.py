"""Contract versioning.

Every envelope that crosses a process boundary carries ``CONTRACT_VERSION``. The docs require
versioning from the start (``ARCHITECTURE.md:106``): a component that receives a version it does
not understand must refuse it rather than silently misparse it.

The version is a single integer, not semver-with-prerelease: these contracts are consumed by
services in a deployment we control, so there is no ecosystem of third-party readers to keep
compatible. A breaking change is a major bump and a coordinated redeploy.
"""

from __future__ import annotations

CONTRACT_VERSION = 1

#: Bumped independently of the schema number so a compatibility shim can be identified.
CONTRACT_REVISION = "1.0.0"


def supports(version: int) -> bool:
    """Return whether a peer speaking ``version`` can be understood by this build."""
    return version == CONTRACT_VERSION