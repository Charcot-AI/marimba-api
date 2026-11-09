import { afterEach, expect, it, vi } from 'vitest';
import { drainSocket } from './socket';

afterEach(() => vi.useRealTimers());

function connection() {
  return Object.assign(new EventTarget(), {
    readyState: WebSocket.OPEN,
    send: vi.fn(),
    close: vi.fn(),
  });
}

it('waits for model completion instead of disconnecting after two seconds', async () => {
  vi.useFakeTimers();
  const socket = connection();
  const pending = drainSocket(socket as unknown as WebSocket, 30000);
  expect(socket.send).toHaveBeenCalledWith('{"type":"stop"}');
  await vi.advanceTimersByTimeAsync(3000);
  expect(socket.close).not.toHaveBeenCalled();
  socket.dispatchEvent(new Event('close'));
  expect(await pending).toBe(true);
  expect(vi.getTimerCount()).toBe(0);
});

it('bounds the wait and releases listeners after a stalled model', async () => {
  vi.useFakeTimers();
  const socket = connection();
  const pending = drainSocket(socket as unknown as WebSocket, 30000);
  await vi.advanceTimersByTimeAsync(30000);
  expect(await pending).toBe(false);
  expect(socket.close).toHaveBeenCalledOnce();
  expect(vi.getTimerCount()).toBe(0);
});

it('cleans up when the connection fails during stop', async () => {
  vi.useFakeTimers();
  const socket = connection();
  socket.send.mockImplementation(() => {
    throw new Error('Synthetic connection failure');
  });
  expect(await drainSocket(socket as unknown as WebSocket, 30000)).toBe(false);
  expect(socket.close).toHaveBeenCalledOnce();
  expect(vi.getTimerCount()).toBe(0);
});
