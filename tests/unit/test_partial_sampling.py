from flexmoe.runtime.partial_staging import should_sample_upload


def test_sampling_visits_each_candidate_layer_and_slot() -> None:
    # A 32-upload stride aliases the tested cyclic schedules and observes
    # only one layer/slot. Any replacement must visit the entire cycle.
    for offload_count in (2, 4, 6, 8):
        visited = {
            index % offload_count
            for index in range(256)
            if should_sample_upload(index, capacity=128, pending_count=0)
        }
        assert visited == set(range(offload_count))


def test_timing_backpressure_never_allocates_unbounded_events() -> None:
    assert not should_sample_upload(0, capacity=0, pending_count=0)
    assert not should_sample_upload(0, capacity=4, pending_count=4)
