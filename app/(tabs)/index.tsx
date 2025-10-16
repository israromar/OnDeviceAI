import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { Colors } from '@/constants/theme';
import * as FileSystem from 'expo-file-system';
import React, { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import {
  Message,
  QWEN3_1_7B_QUANTIZED,
  useExecutorchModule,
  useLLM,
} from 'react-native-executorch';

const MODEL_URL = 'https://your.host/m2m100_418M_int8.pte';
const MODEL_PATH = FileSystem.documentDirectory + 'm2m100_418M_int8.pte';

function useDownloadedModel() {
  const [modelPath, setModelPath] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const info = await FileSystem.getInfoAsync(MODEL_PATH);
        if (!info.exists) {
          await FileSystem.downloadAsync(MODEL_URL, MODEL_PATH);
        }
        setModelPath(MODEL_PATH);
      } catch (e) {
        console.warn('Model download failed', e);
        setModelPath(null);
      }
    })();
  }, []);

  return modelPath;
}

export default function ChatScreen() {
  const llm = useLLM({ model: QWEN3_1_7B_QUANTIZED, preventLoad: Platform.OS === 'web' });
  const modelPath = useDownloadedModel();
  const executorchModule = useExecutorchModule({
    modelSource: modelPath ? { uri: modelPath } : undefined,
  });
  console.log('🚀 ~ ChatScreen ~ executorchModule:', executorchModule);

  const [input, setInput] = useState('Hello');

  //   const configuredRef = useRef(false);
  // console.log('🚀 ~ ChatScreen ~ llm:', llm);x

  //   useEffect(() => {
  //     if (configuredRef.current) return;
  //     configuredRef.current = true;
  //     llm.configure({
  //       chatConfig: {
  //         systemPrompt:
  //           "You are an offline medical translator between doctor and patient. If a message starts with 'ur:', translate to Urdu. If it starts with 'en:', translate to English. Otherwise, detect and translate to the other party's language. Keep responses concise, accurate, and empathetic. Do not provide diagnoses.",
  //         initialMessageHistory: [
  //           {
  //             role: 'assistant',
  //             content: "I'm ready to translate. Prefix with 'en:' for English or 'ur:' for Urdu.",
  //           },
  //         ],
  //       },
  //     });
  //     return () => {
  //       if (llm.isGenerating) {
  //         llm.interrupt();
  //       }
  //     };
  //   }, [llm]);

  // Ensure safe unmount
  useEffect(() => {
    return () => {
      if (llm.isGenerating) {
        llm.interrupt();
      }
    };
  }, [llm]);

  const onSend = useCallback(async () => {
    if (!input.trim() || !llm.isReady || llm.isGenerating) return;
    const text = input.trim();
    setInput('');

    const message = 'Hi, who are you?';
    await llm.sendMessage(message);
    await llm.sendMessage(text);
  }, [input, llm]);

  const handleGenerate = () => {
    const chat: Message[] = [
      { role: 'system', content: 'You are a helpful assistant' },
      { role: 'user', content: 'Hi!' },
      { role: 'assistant', content: 'Hi!, how can I help you?' },
      { role: 'user', content: 'What is the meaning of life?' },
    ];

    // Chat completion
    llm.generate(chat);
  };

  const onStop = useCallback(() => {
    if (llm.isGenerating) llm.interrupt();
  }, [llm]);

  console.log('🚀 ~ ChatScreen ~ llm.downloadProgress:', {
    isGenerating: llm.isGenerating,
    response: llm.response,
  });
  return (
    <KeyboardAvoidingView
      behavior={Platform.select({ ios: 'padding', android: 'height' })}
      style={{ flex: 1, marginTop: 30 }}
    >
      <ThemedView style={styles.container}>
        <ThemedText type='title' style={styles.title}>
          On-device Chat
        </ThemedText>

        {Platform.OS === 'web' ? (
          <View style={styles.center}>
            <Text style={styles.bubbleText}>
              ExecuTorch is not available on web. Build a dev client for iOS/Android.
            </Text>
          </View>
        ) : !llm.isReady ? (
          <View style={styles.center}>
            <ActivityIndicator size='large' />
            <Text style={styles.progressText}>
              Downloading model {Math.round((llm.downloadProgress || 0) * 100)}%
            </Text>
            <View style={styles.progressBar}>
              <View
                style={[
                  styles.progressFill,
                  { width: `${Math.round((llm.downloadProgress || 0) * 100)}%` },
                ]}
              />
            </View>
            {llm.error ? <Text style={styles.error}>{String(llm.error)}</Text> : null}
          </View>
        ) : (
          <View style={styles.messages}>
            {llm.messageHistory.map((m, idx) => (
              <View
                key={idx}
                style={[styles.bubble, m.role === 'user' ? styles.user : styles.assistant]}
              >
                <Text
                  style={[
                    styles.bubbleText,
                    m.role === 'user' ? styles.userText : styles.assistantText,
                  ]}
                >
                  {m.content}
                </Text>
              </View>
            ))}
            {llm.isGenerating ? (
              <View style={[styles.bubble, styles.assistant]}>
                <Text style={[styles.bubbleText, styles.assistantText]}>{llm.response || '…'}</Text>
              </View>
            ) : null}
          </View>
        )}

        <View style={styles.inputRow}>
          <TextInput
            style={styles.input}
            value={input}
            onChangeText={setInput}
            placeholderTextColor={Colors.light.icon}
            placeholder="Prefix with 'ur:' for Urdu or 'en:' for English"
            editable={llm.isReady && !llm.isGenerating}
          />
          <Pressable
            onPress={llm.isGenerating ? onStop : onSend}
            style={[styles.button, llm.isGenerating ? styles.stop : styles.send]}
          >
            <Text style={styles.buttonText}>{llm.isGenerating ? 'Stop' : 'Send'}</Text>
          </Pressable>
        </View>
      </ThemedView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    gap: 12,
  },
  title: {
    textAlign: 'center',
  },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  progressText: {
    fontSize: 14,
    textAlign: 'center',
    color: '#fff',
  },
  progressBar: {
    width: '80%',
    height: 8,
    borderRadius: 6,
    overflow: 'hidden',
    backgroundColor: '#E0E0E0',
    marginTop: 8,
  },
  progressFill: {
    height: '100%',
    backgroundColor: Colors.light.tint,
  },
  error: {
    color: '#b00020',
  },
  messages: {
    flex: 1,
    gap: 8,
  },
  bubble: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 12,
    maxWidth: '90%',
  },
  user: {
    alignSelf: 'flex-end',
    backgroundColor: '#DCF8C6',
  },
  assistant: {
    alignSelf: 'flex-start',
    backgroundColor: '#EEE',
  },
  bubbleText: {
    fontSize: 16,
    color: '#fff',
    textAlign: 'center',
    fontWeight: 'bold',
  },
  inputRow: {
    flexDirection: 'row',
    gap: 8,
    alignItems: 'center',
    paddingBottom: Platform.select({ ios: 16, android: 8 }),
  },
  input: {
    flex: 1,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: '#CCC',
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 10,
    backgroundColor: 'white',
  },
  button: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 12,
  },
  send: {
    backgroundColor: Colors.light.tint,
  },
  stop: {
    backgroundColor: '#FF6B6B',
  },
  buttonText: {
    color: 'white',
    fontWeight: '600',
  },
  userText: {
    color: '#000',
  },
  assistantText: {
    color: '#000',
  },
});
