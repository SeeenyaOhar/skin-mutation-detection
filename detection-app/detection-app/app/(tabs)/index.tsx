import { Modal, StyleSheet, View } from 'react-native';
import { Camera, Point, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import { useRef, useState, useEffect, useCallback } from 'react';
import PermissionsPage from '@/components/PermissionsPage';
// import { loadTensorflowModel, useTensorflowModel } from 'react-native-fast-tflite';
import { TouchableOpacity,  Text  } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useAssets } from 'expo-asset';
import { preprocessImage } from '@/components/preprocess';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';
import { Gesture, GestureDetector } from 'react-native-gesture-handler'
import { runOnJS } from 'react-native-reanimated';
  import { useRouter, useLocalSearchParams } from 'expo-router';
  import { useTfModel } from '@/components/ModelContext';
import {useTensorflowModel} from 'react-native-fast-tflite';
import { preprocessImageTflite } from '@/components/preprocess_tflite';
// import NoCameraDeviceError from '`@/components/NoCameraDeviceError';

export default function CameraTabScreen() {
  const device = useCameraDevice('back', {
    physicalDevices: ['ultra-wide-angle-camera'],
  });
  const { hasPermission } = useCameraPermission();
  const cameraRef = useRef<Camera>(null);
  const insets = useSafeAreaInsets();
  const [predictionResult, setPredictionResult] = useState<string | null>(null);
  const [showPrediction, setShowPrediction] = useState(false);
  const router = useRouter();

  // const plugin = useTensorflowModel(require('../../assets/model/model2.tflite'))

  if (!hasPermission) return <PermissionsPage />;
//   if (device == null) return <NoCameraDeviceError />;

const takePicture = async () => {
  if (cameraRef.current == null) return;

  try {
    const photo = await cameraRef.current.takePhoto();
    const imagePath = `file://${photo.path}`;

    router.push({
      pathname: '/prediction-result',
      params: {
        photoUri: imagePath,
      },
    });
  } catch (err) {
    console.error('Failed to take photo:', err);
  }
};

//  const takePictureTfLite = async () => {
//   if (cameraRef.current == null) return;

//   try {
//     const photo = await cameraRef.current.takePhoto();
//     const imagePath = `file://${photo.path}`;
//     setPhotoUri(imagePath);

//     const input = await preprocessImageTflite(imagePath);

//     if (plugin?.model) {
//       const outputMap = await plugin.model.run([input]);
//       const scores = Object.values(outputMap)[0] as Float32Array;

//       const maxIndex = scores.indexOf(Math.max(...scores));
//       const predictedLabel = LABELS[maxIndex];
//       const confidence = scores[maxIndex];

//       router.push({
//         pathname: '/prediction-result',
//         params: {
//           label: predictedLabel,
//           confidence: (confidence * 100).toFixed(2),
//           photoUri: imagePath,  // pass photo URI here
//         },
//       });
//     }
//   } catch (err) {
//     console.error('Failed to take photo or run TFLite prediction:', err);
//   }
// };

  const focus = useCallback((point: Point) => {
    console.log("focus");
    return cameraRef.current!.focus(point)
  }, [])

  const gesture = Gesture.Tap()
    .onEnd(({ x, y }) => {
      runOnJS(focus)({ x, y })
    })

  return (
    <View style={styles.container}>
    <GestureDetector gesture={gesture}>
      <Camera
        style={[
          StyleSheet.absoluteFill
        ]}
        device={device!}
        isActive={true}
        photo={true}
        ref={cameraRef}
      />
    </GestureDetector>

      {/* Floating UI Controls */}
      <View style={[styles.controls, { bottom: insets.bottom + 24 }]}  >
      <TouchableOpacity
        onPress={takePicture}
        style={[
          styles.captureButton
        ]}
      >
        <View style={styles.innerButton} />
      </TouchableOpacity>

      </View>
      <Modal
        visible={showPrediction}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setShowPrediction(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalText}>Prediction:</Text>
            <Text style={styles.modalLabel}>{predictionResult}</Text>
            <TouchableOpacity onPress={() => setShowPrediction(false)} style={styles.closeButton}>
              <Text style={styles.closeButtonText}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  headerImage: {
    color: '#808080',
    bottom: -90,
    left: -35,
    position: 'absolute',
  },
  titleContainer: {
    flexDirection: 'row',
    gap: 8,
  },
  cameraContainer: {
    height: 400,
    borderRadius: 16,
    overflow: 'hidden',
    marginVertical: 16,
  },
    container: {
    flex: 1,
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButton: {
    position: 'absolute',
    bottom: 40,
    alignSelf: 'center',
    width: 80,
    height: 80,
    borderRadius: 40,
    borderWidth: 4,
    borderColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
  },
  innerButton: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#fff',
  },
  preview: {
    position: 'absolute',
    bottom: 140,
    right: 20,
    width: 100,
    height: 150,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: '#fff',
  },
   controls: {
    position: 'absolute',
    bottom: 0,
    width: '100%',
    alignItems: 'center',
  },
  modalOverlay: {
  flex: 1,
  backgroundColor: 'rgba(0,0,0,0.6)',
  justifyContent: 'center',
  alignItems: 'center',
  },
  modalContent: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 24,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 5,
  },
  modalText: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  modalLabel: {
    fontSize: 24,
    fontWeight: '600',
    color: '#333',
    marginBottom: 16,
  },
  closeButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 8,
  },
  closeButtonText: {
    color: '#fff',
    fontWeight: 'bold',
  }
});
