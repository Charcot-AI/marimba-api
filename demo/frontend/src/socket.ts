export async function drainSocket(socket: WebSocket | undefined, waitMs: number): Promise<boolean> {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    socket?.close();
    return true;
  }
  return new Promise((resolve) => {
    const finish = (drained: boolean) => {
      clearTimeout(timer);
      socket.removeEventListener('close', onClose);
      resolve(drained);
    };
    const onClose = () => finish(true);
    const timer = setTimeout(() => {
      socket.close();
      finish(false);
    }, waitMs);
    socket.addEventListener('close', onClose, { once: true });
    try {
      socket.send(JSON.stringify({ type: 'stop' }));
    } catch {
      socket.close();
      finish(false);
    }
  });
}
