// components/PermissionsPage.tsx

import React from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';
import { useCameraPermission } from 'react-native-vision-camera';

export default function PermissionsPage() {
  const { hasPermission, requestPermission } = useCameraPermission();

  return (
    <View style={styles.container}>
      <Text style={styles.text}>Camera permission is required.</Text>
      <Button title="Grant Permission" onPress={requestPermission} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  text: {
    marginBottom: 10,
    fontSize: 16,
  },
});
