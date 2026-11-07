import js from '@eslint/js';
import ts from 'typescript-eslint';
export default ts.config(js.configs.recommended, ...ts.configs.recommended, {
  files: ['src/**/*.{ts,js}'],
  languageOptions: {
    globals: {
      console: 'readonly',
      AudioWorkletProcessor: 'readonly',
      registerProcessor: 'readonly',
      sampleRate: 'readonly',
      currentTime: 'readonly',
    },
  },
});
