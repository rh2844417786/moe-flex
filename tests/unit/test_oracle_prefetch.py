from dataclasses import dataclass

import pytest


@dataclass
class Handle:
    number: int


class FakeTransfer:
    def __init__(self, *, ready=True):
        self.ready = ready
        self.enqueued = []
        self.waited_handles = []
        self._serial = 0

    def enqueue(self, copies):
        self._serial += 1
        self.enqueued.append(tuple(copies))
        return Handle(self._serial)

    def query(self, handle):
        return self.ready

    def wait(self, handle):
        self.waited_handles.append(handle)
        self.ready = True

    def elapsed_ms(self, handle):
        return 1.25 if self.ready else None

    def synchronize(self):
        self.ready = True


def test_prefetch_reuses_only_free_existing_ingress_slots():
    from flexmoe.runtime.oracle_prefetch import IngressCapacityError, OracleIngress

    ingress = OracleIngress(
        capacity=4, expert_bytes=24, transfer_backend=FakeTransfer()
    )
    ticket = ingress.schedule(step=3, source_layer=1, target_layer=2, expert_ids=(4, 5))

    assert ticket.slots == (0, 1)
    assert ingress.free_slots == (2, 3)
    with pytest.raises(IngressCapacityError):
        ingress.schedule(step=3, source_layer=1, target_layer=3, expert_ids=(6, 7, 8))


def test_resolve_waits_only_for_required_not_ready_experts():
    from flexmoe.runtime.oracle_prefetch import OracleIngress

    backend = FakeTransfer(ready=False)
    ingress = OracleIngress(capacity=4, expert_bytes=24, transfer_backend=backend)
    ticket = ingress.schedule(step=1, source_layer=0, target_layer=1, expert_ids=(2, 3))

    resolved = ingress.resolve(step=1, layer=1, required_ids=(2, 4))

    assert resolved.prefetched_ids == (2,)
    assert resolved.fallback_ids == (4,)
    assert resolved.ticket == ticket
    assert backend.waited_handles == [ticket.handle]
    assert resolved.ready_before_use_count == 0
    assert resolved.layer_batch_all_ready is False


def test_finish_layer_releases_only_matching_in_use_slots():
    from flexmoe.runtime.oracle_prefetch import OracleIngress

    ingress = OracleIngress(
        capacity=3, expert_bytes=24, transfer_backend=FakeTransfer()
    )
    ingress.schedule(step=1, source_layer=0, target_layer=1, expert_ids=(2, 3))
    ingress.resolve(step=1, layer=1, required_ids=(2, 3))
    ingress.schedule(step=1, source_layer=0, target_layer=2, expert_ids=(1,))

    ingress.finish_layer(step=1, layer=1)

    assert ingress.free_slots == (0, 1)
    assert ingress.entries == {(1, 2, 1): 2}


def test_duplicate_schedule_is_rejected_without_aliasing_slots():
    from flexmoe.runtime.oracle_prefetch import OracleIngress

    ingress = OracleIngress(
        capacity=2, expert_bytes=24, transfer_backend=FakeTransfer()
    )
    ingress.schedule(step=1, source_layer=0, target_layer=1, expert_ids=(2,))

    with pytest.raises(ValueError, match="already scheduled"):
        ingress.schedule(step=1, source_layer=0, target_layer=1, expert_ids=(2,))
    assert ingress.free_slots == (1,)
