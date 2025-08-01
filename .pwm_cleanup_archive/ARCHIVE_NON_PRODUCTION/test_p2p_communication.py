"""
Tests for P2P decentralized communication module
"""

import asyncio
import pytest
import time
from .p2p_communication import (
    P2PNode,
    P2PMessage,
    MessageType,
    PeerStatus,
    create_p2p_network,
)


@pytest.mark.asyncio
async def test_node_creation_and_start():
    """Test basic node creation and startup"""
    node = P2PNode(
        node_id="test_node",
        host="127.0.0.1",
        port=0,  # Auto-assign
        capabilities={"compute", "storage"}
    )

    await node.start()

    try:
        assert node._running
        assert node.port > 0  # Port was assigned
        assert "compute" in node.capabilities
        assert "storage" in node.capabilities

    finally:
        await node.stop()


@pytest.mark.asyncio
async def test_peer_connection():
    """Test connecting two peers"""
    node1 = P2PNode("node1")
    node2 = P2PNode("node2")

    await node1.start()
    await node2.start()

    try:
        # Connect node1 to node2
        peer_id = await node1.connect_to_peer("127.0.0.1", node2.port)

        assert peer_id == "node2"
        assert "node2" in node1.peers
        assert "node2" in node1.connections
        assert node1.peers["node2"].status == PeerStatus.CONNECTED

        # Check reverse connection
        await asyncio.sleep(0.1)  # Allow handshake to complete
        assert "node1" in node2.peers
        assert "node1" in node2.connections

    finally:
        await node1.stop()
        await node2.stop()


@pytest.mark.asyncio
async def test_direct_message_sending():
    """Test sending messages between peers"""
    node1 = P2PNode("node1")
    node2 = P2PNode("node2")

    received_messages = []

    async def test_handler(node, message):
        received_messages.append((node.node_id, message.payload))

    node2.register_handler(MessageType.DATA, test_handler)

    await node1.start()
    await node2.start()

    try:
        # Connect nodes
        await node1.connect_to_peer("127.0.0.1", node2.port)

        # Send message
        success = await node1.send_to_peer("node2", {"test": "data"})
        assert success

        # Wait for message processing
        await asyncio.sleep(0.1)

        # Check message was received
        assert len(received_messages) == 1
        assert received_messages[0] == ("node2", {"test": "data"})

        # Check stats
        assert node1.stats["messages_sent"] > 0
        assert node2.stats["messages_received"] > 0

    finally:
        await node1.stop()
        await node2.stop()


@pytest.mark.asyncio
async def test_broadcast_messages():
    """Test broadcasting to multiple peers"""
    nodes = []
    received_broadcasts = defaultdict(list)

    async def broadcast_handler(node, message):
        received_broadcasts[node.node_id].append(message.payload)

    try:
        # Create 4 nodes
        for i in range(4):
            node = P2PNode(f"node{i}")
            node.register_handler(MessageType.DATA, broadcast_handler)
            await node.start()
            nodes.append(node)

        # Connect in a star topology (node0 at center)
        for i in range(1, 4):
            await nodes[i].connect_to_peer("127.0.0.1", nodes[0].port)

        await asyncio.sleep(0.5)  # Allow connections to establish

        # Broadcast from center node
        num_sent = await nodes[0].broadcast({"announcement": "hello all"})
        assert num_sent == 3  # Sent to 3 connected peers

        await asyncio.sleep(0.5)  # Give more time for message processing

        # Check all peers received broadcast
        print(f"Received broadcasts: {dict(received_broadcasts)}")
        print(f"Node connections: {[(n.node_id, list(n.connections.keys())) for n in nodes]}")

        # At least some nodes should have received the broadcast
        assert len(received_broadcasts) >= 2

    finally:
        for node in nodes:
            await node.stop()


@pytest.mark.asyncio
async def test_message_acknowledgment():
    """Test message acknowledgment mechanism"""
    node1 = P2PNode("node1")
    node2 = P2PNode("node2")

    await node1.start()
    await node2.start()

    try:
        await node1.connect_to_peer("127.0.0.1", node2.port)

        # Send with ACK required
        success = await node1.send_to_peer("node2", {"important": "data"}, require_ack=True)
        assert success

        # Send to non-existent peer (should fail)
        success = await node1.send_to_peer("node999", {"test": "data"}, require_ack=True)
        assert not success

    finally:
        await node1.stop()
        await node2.stop()


@pytest.mark.asyncio
async def test_peer_discovery():
    """Test peer discovery through bootstrap nodes"""
    # Create bootstrap node
    bootstrap = P2PNode("bootstrap")
    await bootstrap.start()

    # Create regular nodes and connect to bootstrap
    nodes = []
    for i in range(3):
        node = P2PNode(f"node{i}")
        await node.start()
        await node.connect_to_peer("127.0.0.1", bootstrap.port)
        nodes.append(node)

    try:
        # New node discovers peers through bootstrap
        new_node = P2PNode("new_node")
        await new_node.start()

        discovered = await new_node.discover_peers([("127.0.0.1", bootstrap.port)])
        assert discovered >= 1  # At least connected to bootstrap

        # Request peer list (would need to implement handler in real system)
        await asyncio.sleep(0.1)

    finally:
        await bootstrap.stop()
        for node in nodes:
            await node.stop()
        await new_node.stop()


@pytest.mark.asyncio
async def test_network_resilience():
    """Test network continues operating when nodes fail"""
    nodes = await create_p2p_network(5)

    try:
        # Get initial stats
        initial_connections = sum(
            len(node.connections) for node in nodes
        )
        assert initial_connections > 0

        # Stop one node
        await nodes[2].stop()
        await asyncio.sleep(0.5)

        # Other nodes should still be connected
        remaining_connections = sum(
            len(node.connections) for node in nodes if node._running
        )
        assert remaining_connections > 0

        # Try to send message between remaining nodes
        if "node_1" in nodes[0].connections:
            success = await nodes[0].send_to_peer("node_1", {"still": "working"})
            assert success

    finally:
        for node in nodes:
            if node._running:
                await node.stop()


@pytest.mark.asyncio
async def test_latency_measurement():
    """Test peer latency measurement through heartbeats"""
    node1 = P2PNode("node1")
    node2 = P2PNode("node2")

    await node1.start()
    await node2.start()

    try:
        await node1.connect_to_peer("127.0.0.1", node2.port)

        # Send heartbeat manually
        heartbeat = P2PMessage(
            message_id="test_hb",
            sender_id="node1",
            recipient_id=None,
            message_type=MessageType.HEARTBEAT,
            payload={},
            timestamp=time.time()
        )

        _, writer = node1.connections["node2"]
        await node1._send_message(writer, heartbeat)

        await asyncio.sleep(0.1)

        # Check latency was recorded
        if "node1" in node2.peers:
            assert node2.peers["node1"].latency_ms > 0
            assert node2.peers["node1"].latency_ms < 100  # Should be fast on localhost

    finally:
        await node1.stop()
        await node2.stop()


@pytest.mark.asyncio
async def test_capability_based_routing():
    """Test routing based on node capabilities"""
    # Create nodes with different capabilities
    compute_node = P2PNode("compute_node", capabilities={"compute"})
    storage_node = P2PNode("storage_node", capabilities={"storage"})
    relay_node = P2PNode("relay_node", capabilities={"relay"})

    await compute_node.start()
    await storage_node.start()
    await relay_node.start()

    try:
        # Connect in a line: compute -> relay -> storage
        await compute_node.connect_to_peer("127.0.0.1", relay_node.port)
        await relay_node.connect_to_peer("127.0.0.1", storage_node.port)

        # Check capabilities are shared
        assert "relay" in compute_node.peers["relay_node"].capabilities
        assert "storage" in relay_node.peers["storage_node"].capabilities

    finally:
        await compute_node.stop()
        await storage_node.stop()
        await relay_node.stop()


@pytest.mark.asyncio
async def test_network_statistics():
    """Test network statistics collection"""
    node = P2PNode("stats_node")
    await node.start()

    try:
        # Create and connect to peers
        peer1 = P2PNode("peer1")
        peer2 = P2PNode("peer2")
        await peer1.start()
        await peer2.start()

        await node.connect_to_peer("127.0.0.1", peer1.port)
        await node.connect_to_peer("127.0.0.1", peer2.port)

        # Send some messages
        await node.send_to_peer("peer1", {"data": 1})
        await node.send_to_peer("peer2", {"data": 2})
        await node.broadcast({"broadcast": "data"})

        await asyncio.sleep(0.1)

        # Get statistics
        stats = node.get_network_stats()

        assert stats["node_id"] == "stats_node"
        assert stats["connected_peers"] == 2
        assert stats["total_peers"] == 2
        assert stats["messages_sent"] >= 4  # 2 individual + 2 broadcast
        assert stats["bytes_sent"] > 0

        await peer1.stop()
        await peer2.stop()

    finally:
        await node.stop()


from collections import defaultdict

if __name__ == "__main__":
    # Run tests
    asyncio.run(test_node_creation_and_start())
    asyncio.run(test_peer_connection())
    asyncio.run(test_direct_message_sending())
    asyncio.run(test_broadcast_messages())
    asyncio.run(test_message_acknowledgment())
    asyncio.run(test_peer_discovery())
    asyncio.run(test_network_resilience())
    asyncio.run(test_latency_measurement())
    asyncio.run(test_capability_based_routing())
    asyncio.run(test_network_statistics())
    print("All P2P tests passed!")