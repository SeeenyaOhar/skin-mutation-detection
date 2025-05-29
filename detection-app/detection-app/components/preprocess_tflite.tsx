import * as ImageManipulator from 'expo-image-manipulator';
import * as FileSystem from 'expo-file-system';
const IMG_SIZE = 456; // your model input size

export async function preprocessImageTflite(uri: string): Promise<Float32Array> {
  // Resize image, output base64 PNG
  const resized = await ImageManipulator.manipulateAsync(
    uri,
    [{ resize: { width: IMG_SIZE, height: IMG_SIZE } }],
    { format: ImageManipulator.SaveFormat.PNG, base64: true }
  );

  const base64 = resized.base64!;
  const rawBinary = atob(base64);

  // Convert base64 string to Uint8Array (RGBA)
  const rgbaArray = new Uint8Array(rawBinary.length);
  for (let i = 0; i < rawBinary.length; i++) {
    rgbaArray[i] = rawBinary.charCodeAt(i);
  }

  const pixelCount = IMG_SIZE * IMG_SIZE;
  const floatArray = new Float32Array(pixelCount * 3);

  for (let i = 0, j = 0; i < pixelCount * 4; i += 4, j += 3) {
    floatArray[j] = (rgbaArray[i]);       // R
    floatArray[j + 1] = (rgbaArray[i + 1]); // G
    floatArray[j + 2] = (rgbaArray[i + 2]); // B
  }

  return floatArray;
}