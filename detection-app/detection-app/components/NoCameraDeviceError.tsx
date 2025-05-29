// components/NoCameraDeviceError.tsx

import { StyleSheet, View, Text } from 'react-native';
import React from 'react';
import { ThemedText } from './ThemedText';
import { ThemedView } from './ThemedView';

export default function NoCameraDeviceError() {
  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title">No Camera Found</ThemedText>
      <ThemedText>
        We couldn't detect any available camera device on this phone.
      </ThemedText>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 24,
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});
