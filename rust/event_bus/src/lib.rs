use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{mpsc, RwLock};
use tokio::task::JoinHandle;
use uuid::Uuid;

/// Event structure mirroring the Python implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub event_id: Uuid,
    pub event_type: String,
    pub payload: Value,
    pub source: Option<String>,
    pub timestamp: f64,
}

impl Event {
    pub fn new(event_type: impl Into<String>, payload: Value, source: Option<String>) -> Self {
        Self {
            event_id: Uuid::new_v4(),
            event_type: event_type.into(),
            payload,
            source,
            timestamp: chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
        }
    }
}

pub struct Subscription {
    pub id: Uuid,
    pub receiver: mpsc::UnboundedReceiver<Event>,
}

pub struct EventBus {
    subscribers: RwLock<HashMap<String, Vec<(Uuid, mpsc::UnboundedSender<Event>)>>>,
    queue_tx: mpsc::UnboundedSender<Event>,
    queue_rx: tokio::sync::Mutex<Option<mpsc::UnboundedReceiver<Event>>>,
    worker: RwLock<Option<JoinHandle<()>>>,
}

impl EventBus {
    pub fn new() -> Arc<Self> {
        let (tx, rx) = mpsc::unbounded_channel();
        Arc::new(Self {
            subscribers: RwLock::new(HashMap::new()),
            queue_tx: tx,
            queue_rx: tokio::sync::Mutex::new(Some(rx)),
            worker: RwLock::new(None),
        })
    }

    pub async fn start(self: &Arc<Self>) {
        let mut rx_opt = self.queue_rx.lock().await;
        if rx_opt.is_none() {
            return;
        }
        let mut rx = rx_opt.take().unwrap();
        let bus = Arc::clone(self);
        let handle = tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                let subscribers = bus.subscribers.read().await;
                if let Some(list) = subscribers.get(&event.event_type) {
                    for (_, tx) in list {
                        let _ = tx.send(event.clone());
                    }
                }
            }
        });
        *self.worker.write().await = Some(handle);
    }

    pub async fn subscribe(self: &Arc<Self>, event_type: &str) -> Subscription {
        let (tx, rx) = mpsc::unbounded_channel();
        let id = Uuid::new_v4();
        let mut subs = self.subscribers.write().await;
        subs.entry(event_type.to_string())
            .or_default()
            .push((id, tx));
        Subscription { id, receiver: rx }
    }

    pub async fn unsubscribe(self: &Arc<Self>, event_type: &str, id: Uuid) {
        let mut subs = self.subscribers.write().await;
        if let Some(list) = subs.get_mut(event_type) {
            list.retain(|(sid, _)| *sid != id);
        }
    }

    pub async fn publish(self: &Arc<Self>, event_type: &str, payload: Value, source: Option<String>) {
        let event = Event::new(event_type, payload, source);
        let _ = self.queue_tx.send(event);
    }

    pub async fn stop(self: &Arc<Self>) {
        if let Some(handle) = self.worker.write().await.take() {
            handle.abort();
            let _ = handle.await;
        }
        let mut lock = self.queue_rx.lock().await;
        if lock.is_none() {
            let (_tx, rx) = mpsc::unbounded_channel();
            *lock = Some(rx);
        }
    }
}

use once_cell::sync::Lazy;

static GLOBAL_BUS: Lazy<RwLock<Option<Arc<EventBus>>>> = Lazy::new(|| RwLock::new(None));

pub async fn get_global_event_bus() -> Arc<EventBus> {
    let mut guard = GLOBAL_BUS.write().await;
    if let Some(bus) = guard.clone() {
        return bus;
    }
    let bus = EventBus::new();
    bus.start().await;
    *guard = Some(bus.clone());
    bus
}

pub async fn reset_global_event_bus() {
    let mut guard = GLOBAL_BUS.write().await;
    if let Some(bus) = guard.take() {
        bus.stop().await;
    }
}


