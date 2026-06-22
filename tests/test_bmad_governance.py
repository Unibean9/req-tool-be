"""Tests for BMAD governance gates (addendum §18)."""

import pytest

from app.graphs.policy import (
    ApprovalRequired,
    GovernanceDenied,
    accept_implementation_readiness,
    finalize_prd,
    force_full_bmad_lifecycle,
    lock_scope,
)

pytestmark = pytest.mark.asyncio


async def test_finalize_prd_requires_approval():
    with pytest.raises(ApprovalRequired):
        await finalize_prd()


async def test_lock_scope_requires_approval():
    with pytest.raises(ApprovalRequired):
        await lock_scope()


async def test_accept_implementation_readiness_requires_approval():
    with pytest.raises(ApprovalRequired):
        await accept_implementation_readiness()


async def test_full_lifecycle_force_raises_governance_denied():
    with pytest.raises(GovernanceDenied):
        await force_full_bmad_lifecycle()
