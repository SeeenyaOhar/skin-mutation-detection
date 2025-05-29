import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import * as FileSystem from 'expo-file-system';
import { decodeJpeg } from '@tensorflow/tfjs-react-native';

const IMG_SIZE = 456; // match your model input size

export async function preprocessImage(uri: string): Promise<tf.Tensor4D> {
  // Read image as base64
  const base64 = await FileSystem.readAsStringAsync(uri, {
    encoding: FileSystem.EncodingType.Base64,
  });

  // Convert base64 to Uint8Array
  const rawImageData = tf.util.encodeString(base64, 'base64').buffer;
  const uint8array = new Uint8Array(rawImageData);

  // Decode JPEG
  const imageTensor = decodeJpeg(uint8array);  // You need to import this from '@tensorflow/tfjs-react-native'

  // Resize to model size
  const resized = tf.image.resizeBilinear(imageTensor, [IMG_SIZE, IMG_SIZE]);

  // Normalize image pixels to [-1, 1]
//   const normalized = resized.div(tf.scalar(127.5)).sub(tf.scalar(1));

  // Add batch dimension (for model input shape)
  return resized.expandDims(0) as tf.Tensor4D;
}