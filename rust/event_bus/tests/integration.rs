use event_bus::{EventBus, get_global_event_bus, reset_global_event_bus};
use serde_json::json;
use std::sync::Arc;
use tokio::time::sleep;
use std::time::Duration;

#[tokio::test]
async fn test_publish_subscribe() {
    let bus = EventBus::new();
    bus.start().await;

    let mut sub = bus.subscribe("test_event").await;
    bus.publish("test_event", json!({"data": "test"}), None).await;
    sleep(Duration::from_millis(10)).await;

    if let Some(event) = sub.receiver.recv().await {
        assert_eq!(event.event_type, "test_event");
        assert_eq!(event.payload["data"], "test");
    } else {
        panic!("no event received");
    }

    bus.stop().await;
}

#[tokio::test]
async fn test_unsubscribe() {
    let bus = EventBus::new();
    bus.start().await;

    let mut sub = bus.subscribe("test_event").await;
    let id = sub.id;
    bus.unsubscribe("test_event", id).await;
    bus.publish("test_event", json!({"data": "test"}), None).await;
    sleep(Duration::from_millis(10)).await;

    assert!(sub.receiver.try_recv().is_err());

    bus.stop().await;
}

#[tokio::test]
async fn test_get_global_event_bus() {
    let bus1 = get_global_event_bus().await;
    let bus2 = get_global_event_bus().await;
    assert!(Arc::ptr_eq(&bus1, &bus2));
    bus1.stop().await;
    reset_global_event_bus().await;
}
