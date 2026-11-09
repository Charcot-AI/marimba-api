export interface Transcript {
  text: string;
  final: boolean;
  id?: string;
  replaces?: string[];
}
export interface SttAdapter {
  start(
    onTranscript: (event: Transcript) => void,
    onError: (message: string) => void,
  ): Promise<void>;
  push(pcm: Int16Array, capturedAtEpochMs: number): void;
  stop(): Promise<void>;
}
export type SttFactory = (config: Record<string, unknown>) => SttAdapter;
