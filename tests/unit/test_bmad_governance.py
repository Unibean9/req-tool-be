"""Tests for BMAD governance gates (addendum §18)."""

import pytest

from app.graphs.policy import (
    ApprovalRequired,
    finalize_prd,
    lock_scope,
)

pytestmark = pytest.mark.asyncio


async def test_finalize_prd_requires_approval():
    with pytest.raises(ApprovalRequired):
        await finalize_prd()


async def test_lock_scope_requires_approval():
    with pytest.raises(ApprovalRequired):
        await lock_scope()
