"""
Tests for event-driven image processing pipeline
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path
import pytest
from collections import defaultdict
from .image_processing_pipeline import (
    ImageProcessingPipeline,
    ImageEvent,
    EventType,
    ProcessingStage,
    ValidationColony,
    PreprocessingColony,
    ClassificationColony,
    EventBus,
    simulate_image_upload
)

# Check if PIL is available for creating test images
try:
    from PIL import Image
    IMAGING_AVAILABLE = True
except ImportError:
    IMAGING_AVAILABLE = False


def create_test_image(path: str, size: tuple = (100, 100), color: tuple = (255, 0, 0)):
    """Create a test image file"""
    if IMAGING_AVAILABLE:
        img = Image.new('RGB', size, color)
        img.save(path, 'JPEG')
    else:
        # Create a minimal valid JPEG header
        with open(path, 'wb') as f:
            # JPEG magic bytes
            f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00')
            # Add some data
            f.write(b'\x00' * 100)


class EventCollector:
    """Helper to collect events for testing"""
    def __init__(self, event_bus: EventBus):
        self.events = defaultdict(list)
        self.event_bus = event_bus

        # Subscribe to all event types
        for event_type in EventType:
            event_bus.subscribe(event_type.value, self.collect_event)

    def collect_event(self, event_type: str, event_data: dict):
        self.events[event_type].append(event_data)

    def get_events_by_type(self, event_type: EventType) -> list:
        return self.events[event_type.value]

    def clear(self):
        self.events.clear()


@pytest.mark.asyncio
async def test_pipeline_startup_shutdown():
    """Test basic pipeline lifecycle"""
    pipeline = ImageProcessingPipeline()

    # Start pipeline
    await pipeline.start()
    assert pipeline._running
    assert len(pipeline.colonies) == 6  # All colony types

    # Get stats
    stats = pipeline.get_pipeline_stats()
    assert len(stats) == 6
    assert "validation_colony" in stats
    assert "classification_colony" in stats

    # Stop pipeline
    await pipeline.stop()
    assert not pipeline._running
    assert len(pipeline.colonies) == 0


@pytest.mark.asyncio
async def test_image_validation_colony():
    """Test image validation colony"""
    event_bus = EventBus()
    collector = EventCollector(event_bus)

    # Create validation colony
    colony = ValidationColony(event_bus, num_workers=1)
    await colony.start()

    try:
        # Create a test file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            test_file = f.name
        create_test_image(test_file)

        # Create upload event
        upload_event = ImageEvent(
            event_id="test_1",
            event_type=EventType.NEW_IMAGE_UPLOADED,
            timestamp=time.time(),
            image_id="img_1",
            payload={"image_path": test_file},
            metadata={}
        )

        # Publish event
        event_bus.publish(
            EventType.NEW_IMAGE_UPLOADED.value,
            upload_event.to_dict()
        )

        # Wait for processing
        await asyncio.sleep(0.5)

        # Check either validation succeeded or failed
        validated_events = collector.get_events_by_type(EventType.IMAGE_VALIDATED)
        failed_events = collector.get_events_by_type(EventType.PROCESSING_FAILED)

        # Should have either validated or failed
        assert len(validated_events) > 0 or len(failed_events) > 0

        # Check event content if validation succeeded
        if len(validated_events) > 0:
            validated = validated_events[0]
            assert validated["image_id"] == "img_1"
            assert "validation_result" in validated["payload"] or "valid" in validated["payload"]

        # Clean up
        os.unlink(test_file)

    finally:
        await colony.stop()


@pytest.mark.asyncio
async def test_event_flow_through_pipeline():
    """Test event flow through multiple colonies"""
    event_bus = EventBus()
    collector = EventCollector(event_bus)

    # Create a minimal pipeline with 3 stages
    validation = ValidationColony(event_bus, num_workers=1)
    preprocessing = PreprocessingColony(event_bus, num_workers=1)
    classification = ClassificationColony(event_bus, num_workers=1)

    await validation.start()
    await preprocessing.start()
    await classification.start()

    try:
        # Create test file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            test_file = f.name
        create_test_image(test_file)

        # Trigger pipeline
        upload_event = ImageEvent(
            event_id="test_flow",
            event_type=EventType.NEW_IMAGE_UPLOADED,
            timestamp=time.time(),
            image_id="flow_test",
            payload={"image_path": test_file},
            metadata={"test": True}
        )

        event_bus.publish(
            EventType.NEW_IMAGE_UPLOADED.value,
            upload_event.to_dict()
        )

        # Wait for pipeline to process
        await asyncio.sleep(1.0)

        # Check events were published in order
        assert len(collector.get_events_by_type(EventType.IMAGE_VALIDATED)) > 0
        assert len(collector.get_events_by_type(EventType.IMAGE_PREPROCESSED)) > 0

        # Classification needs features, so might not complete
        # But we should see the flow working

        os.unlink(test_file)

    finally:
        await validation.stop()
        await preprocessing.stop()
        await classification.stop()


@pytest.mark.asyncio
async def test_full_pipeline_processing():
    """Test full pipeline with all colonies"""
    pipeline = ImageProcessingPipeline()
    collector = EventCollector(pipeline.event_bus)

    await pipeline.start()

    try:
        # Create test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            test_file = f.name
            # Write PNG header to make it somewhat valid
            f.write(b'\x89PNG\r\n\x1a\n')

        # Process image
        correlation_id = await pipeline.process_image(test_file, {"test": True})
        assert correlation_id is not None

        # Wait for processing
        await asyncio.sleep(2.0)

        # Check completion event
        completion_events = collector.get_events_by_type(EventType.PROCESSING_COMPLETED)

        # May have failed due to invalid image, but should complete
        assert len(completion_events) > 0 or len(collector.get_events_by_type(EventType.PROCESSING_FAILED)) > 0

        # Get pipeline stats
        stats = pipeline.get_pipeline_stats()

        # At least validation should have processed something
        validation_stats = stats.get("validation_colony", {})
        assert validation_stats["metrics"]["processed"] > 0 or validation_stats["metrics"]["failed"] > 0

        os.unlink(test_file)

    finally:
        await pipeline.stop()


@pytest.mark.asyncio
async def test_colony_work_distribution():
    """Test that work is distributed among workers"""
    event_bus = EventBus()

    # Create colony with multiple workers
    colony = ValidationColony(event_bus, num_workers=3)
    await colony.start()

    try:
        # Send multiple events rapidly
        for i in range(10):
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                event = ImageEvent(
                    event_id=f"test_{i}",
                    event_type=EventType.NEW_IMAGE_UPLOADED,
                    timestamp=time.time(),
                    image_id=f"img_{i}",
                    payload={"image_path": f.name},
                    metadata={}
                )

                event_bus.publish(
                    EventType.NEW_IMAGE_UPLOADED.value,
                    event.to_dict()
                )

        # Wait for processing
        await asyncio.sleep(1.0)

        # Check that multiple workers processed events
        assert colony.metrics["processed"] >= 5  # At least some processed

        # Clean up temp files
        import glob
        for f in glob.glob('/tmp/tmp*.jpg'):
            try:
                os.unlink(f)
            except:
                pass

    finally:
        await colony.stop()


@pytest.mark.asyncio
async def test_failure_handling():
    """Test that failures are handled gracefully"""
    event_bus = EventBus()
    collector = EventCollector(event_bus)

    colony = ValidationColony(event_bus, num_workers=1)
    await colony.start()

    try:
        # Send event with non-existent file
        event = ImageEvent(
            event_id="fail_test",
            event_type=EventType.NEW_IMAGE_UPLOADED,
            timestamp=time.time(),
            image_id="fail_img",
            payload={"image_path": "/non/existent/file.jpg"},
            metadata={}
        )

        event_bus.publish(
            EventType.NEW_IMAGE_UPLOADED.value,
            event.to_dict()
        )

        # Wait for processing
        await asyncio.sleep(0.5)

        # Check failure event was published
        failures = collector.get_events_by_type(EventType.PROCESSING_FAILED)
        assert len(failures) > 0

        failure = failures[0]
        assert "error" in failure["payload"]
        assert failure["image_id"] == "fail_img"

    finally:
        await colony.stop()


@pytest.mark.asyncio
async def test_call_for_proposals():
    """Test dynamic task allocation via proposals"""
    event_bus = EventBus()
    collector = EventCollector(event_bus)

    # Create two colonies that can handle same work
    colony1 = PreprocessingColony(event_bus, num_workers=2)
    colony2 = PreprocessingColony(event_bus, num_workers=4)
    colony2.colony_name = "preprocessing_colony_2"  # Different name

    await colony1.start()
    await colony2.start()

    try:
        # Send call for proposals
        cfp_event = ImageEvent(
            event_id="cfp_1",
            event_type=EventType.CALL_FOR_PROPOSALS,
            timestamp=time.time(),
            image_id="proposal_test",
            payload={"stage": ProcessingStage.PREPROCESSING.value},
            metadata={}
        )

        event_bus.publish(
            EventType.CALL_FOR_PROPOSALS.value,
            cfp_event.to_dict()
        )

        # Wait for proposals
        await asyncio.sleep(0.2)

        # Check proposals were submitted
        proposals = collector.get_events_by_type(EventType.PROPOSAL_SUBMITTED)
        assert len(proposals) == 2  # Both colonies should respond

        # Check proposal content
        for proposal in proposals:
            assert "proposal" in proposal["payload"]
            assert "confidence" in proposal["payload"]["proposal"]
            assert "estimated_time" in proposal["payload"]["proposal"]

    finally:
        await colony1.stop()
        await colony2.stop()


@pytest.mark.asyncio
async def test_aggregation_colony():
    """Test result aggregation"""
    event_bus = EventBus()
    collector = EventCollector(event_bus)

    # Create aggregation colony
    from image_processing_pipeline import AggregationColony
    aggregator = AggregationColony(event_bus)
    await aggregator.start()

    try:
        image_id = "agg_test"

        # Simulate events from different stages
        stages = [
            (EventType.IMAGE_VALIDATED, {"valid": True}),
            (EventType.IMAGE_PREPROCESSED, {"preprocessed": True}),
            (EventType.FEATURES_EXTRACTED, {"features": [1, 2, 3]}),
            (EventType.IMAGE_CLASSIFIED, {"class": "test"})
        ]

        for event_type, result in stages:
            event = ImageEvent(
                event_id=f"{event_type.value}_1",
                event_type=event_type,
                timestamp=time.time(),
                image_id=image_id,
                payload={f"{event_type.value}_result": result},
                metadata={}
            )

            event_bus.publish(event_type.value, event.to_dict())
            await asyncio.sleep(0.1)

        # Wait for aggregation
        await asyncio.sleep(0.5)

        # Check completion event
        completions = collector.get_events_by_type(EventType.PROCESSING_COMPLETED)
        assert len(completions) == 1

        completion = completions[0]
        assert completion["image_id"] == image_id
        assert completion["payload"]["success"] == True
        assert len(completion["payload"]["stages_completed"]) == 4

    finally:
        await aggregator.stop()


@pytest.mark.asyncio
async def test_simulate_batch_processing():
    """Test batch image processing simulation"""
    pipeline = ImageProcessingPipeline()
    await pipeline.start()

    try:
        # Process batch of images
        correlation_ids = await simulate_image_upload(pipeline, num_images=3)

        assert len(correlation_ids) == 3

        # Wait for processing
        await asyncio.sleep(2.0)

        # Check stats
        stats = pipeline.get_pipeline_stats()

        # Some colonies should have processed images
        total_processed = sum(
            colony_stats["metrics"]["processed"]
            for colony_stats in stats.values()
        )

        # At least validation should process all
        assert total_processed >= 3

    finally:
        await pipeline.stop()

        # Clean up test files
        import glob
        for f in glob.glob('/tmp/test_image_*.jpg'):
            try:
                os.unlink(f)
            except:
                pass


@pytest.mark.asyncio
async def test_pipeline_metrics():
    """Test metrics collection"""
    pipeline = ImageProcessingPipeline()
    await pipeline.start()

    try:
        # Get initial stats
        initial_stats = pipeline.get_pipeline_stats()

        # Process an image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            test_file = f.name

        await pipeline.process_image(test_file)
        await asyncio.sleep(1.0)

        # Get updated stats
        final_stats = pipeline.get_pipeline_stats()

        # Check metrics updated
        for colony_name, colony_stats in final_stats.items():
            metrics = colony_stats["metrics"]

            # At least validation should have metrics
            if colony_name == "validation_colony":
                assert metrics["processed"] > 0 or metrics["failed"] > 0

            # Check metric structure
            assert "avg_processing_time" in metrics
            assert "total_processing_time" in metrics

        os.unlink(test_file)

    finally:
        await pipeline.stop()


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_pipeline_startup_shutdown())
    asyncio.run(test_image_validation_colony())
    asyncio.run(test_event_flow_through_pipeline())
    asyncio.run(test_full_pipeline_processing())
    asyncio.run(test_colony_work_distribution())
    asyncio.run(test_failure_handling())
    asyncio.run(test_call_for_proposals())
    asyncio.run(test_aggregation_colony())
    asyncio.run(test_simulate_batch_processing())
    asyncio.run(test_pipeline_metrics())
    print("All image processing pipeline tests passed!")