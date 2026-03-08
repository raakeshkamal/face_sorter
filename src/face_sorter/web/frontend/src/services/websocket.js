// WebSocket service for real-time progress updates
class WebSocketService {
  constructor() {
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    // Store connection parameters for reconnection
    this.operationType = null;
    this.taskId = null;
    this.onMessage = null;
    this.onError = null;
  }

  connect(operationType, taskId, onMessage, onError) {
    // Store connection parameters for reconnection
    this.operationType = operationType;
    this.taskId = taskId;
    this.onMessage = onMessage;
    this.onError = onError;

    const wsUrl = `ws://127.0.0.1:8000/api/operations/ws/${operationType}/${taskId}`;
    console.log(`[WebSocketService] Connecting to WebSocket: ${wsUrl}`);
    console.log(`[WebSocketService] operationType: ${operationType}, taskId: ${taskId}`);

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log("[WebSocketService] WebSocket connected successfully");
      this.reconnectAttempts = 0;
    };

    this.ws.onmessage = (event) => {
      console.log("[WebSocketService] Received raw message:", event.data);
      try {
        const data = JSON.parse(event.data);
        console.log("[WebSocketService] Parsed message:", data);
        if (onMessage) {
          console.log("[WebSocketService] Calling onMessage callback");
          onMessage(data);
          console.log("[WebSocketService] onMessage callback completed");
        } else {
          console.warn("[WebSocketService] No onMessage callback provided");
        }
      } catch (error) {
        console.error("[WebSocketService] Error parsing WebSocket message:", error);
        console.error("[WebSocketService] Raw message that failed to parse:", event.data);
      }
    };

    this.ws.onerror = (error) => {
      console.error("[WebSocketService] WebSocket error:", error);
      console.error("[WebSocketService] Error details:", JSON.stringify(error));
      if (onError) {
        onError(error);
      }
    };

    this.ws.onclose = (event) => {
      console.log("[WebSocketService] WebSocket closed");
      console.log("[WebSocketService] Close code:", event.code, "reason:", event.reason);
      this.reconnect();
    };
  }

  reconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts && this.taskId) {
      this.reconnectAttempts++;
      console.log(`Reconnecting... attempt ${this.reconnectAttempts}`);
      setTimeout(() => {
        this.connect(this.operationType, this.taskId, this.onMessage, this.onError);
      }, 1000 * this.reconnectAttempts);
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    // Reset connection parameters
    this.operationType = null;
    this.taskId = null;
    this.onMessage = null;
    this.onError = null;
    this.reconnectAttempts = 0;
  }

  isConnected() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }
}

export default new WebSocketService();