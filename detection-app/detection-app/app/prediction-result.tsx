import React, { useEffect, useState } from 'react';
import { View, Text, Image, ActivityIndicator, TouchableOpacity, StyleSheet } from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';
import { preprocessImage } from '@/components/preprocess'; // your preprocess function
import { useTfModel } from '@/components/ModelContext';

const LABELS_UA = [
  'меланоми',
  'родимки',
  'базальноклітинної карциноми',
  'актиничного кератозу',
  'доброякісного кератозу',
  'дерматофіброми',
  'судинної патології',
  'плоскоклітинної карциноми'
];

const RECOMMENDED = [
  true,
  false,
  true,
  true,
  false,
  false,
  true,
  true
]

export default function PredictionResultScreen() {
  const router = useRouter();
  const { photoUri } = useLocalSearchParams();

  const [predicting, setPredicting] = useState(false);
  const [cls, setClass] = useState<number | null>(null);
  const [label, setLabel] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const { model, loading: loadingModel } = useTfModel();
  

    useEffect(() => {
      async function runPrediction() {
        if (!model || !photoUri) return;

        setPredicting(true);
        try {
          const inputTensor = await preprocessImage(photoUri as string);
          const prediction = (await model.predict(inputTensor)) as tf.Tensor;
          const predictionArray = (await prediction.array()) as number[][];
          const scores = predictionArray[0];

          const maxScore = Math.max(...scores);
          const maxIndex = scores.indexOf(maxScore);
          const threshold = 0.25;
          
          if (maxScore >= threshold) {
            setClass(maxIndex);
            setLabel(LABELS_UA[maxIndex]);
            setConfidence(maxScore);
          } else {
            setLabel(null);
            setClass(null);
            setConfidence(maxScore);
          }
        } catch (err) {
          console.error('Failed to run prediction:', err);
        } finally {
          setPredicting(false);
        }
      }

      runPrediction();
    }, [model, photoUri]);

return (
  <View style={styles.container}>
    {loadingModel && <ActivityIndicator size="large" color="#0000ff" />}

    {photoUri && (
      <Image
        source={{ uri: photoUri as string }}
        style={styles.image}
        resizeMode="cover"
      />
    )}

    {!loadingModel && predicting && (
      <ActivityIndicator size="large" color="#007AFF" style={styles.indicator} />
    )}

    {!loadingModel && !predicting && (
      <View style={styles.resultContainer}>
        {label && confidence != null ? (
          <>
            <Text style={styles.label}>Було виявлено ознаки {label}</Text>
              {typeof cls === 'number' && RECOMMENDED?.[cls] ? (
                <Text style={styles.recommendation}>
                  Рекомендуємо звернутися до дерматолога
                </Text>
              )
              : (
                <Text style={styles.recommendation}>
                  Не потребує термінового звернення до лікаря
                </Text>
              )}
          </>
        ) : (
          <Text style={styles.noDetection}>
            Не виявлено ознак уражень/мутацій шкіри
          </Text>
        )}
      </View>
    )}

    <TouchableOpacity style={styles.button} onPress={() => router.back()}>
      <Text style={styles.buttonText}>Повернутися до камери</Text>
    </TouchableOpacity>
  </View>
);

}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000', // black background for full screen image
  },
  image: {
    ...StyleSheet.absoluteFillObject, // full screen
    width: undefined,
    height: undefined,
  },
  indicator: {
    marginVertical: 20,
  },
  resultContainer: {
    position: 'absolute',
    bottom: 100,
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.5)',
    paddingHorizontal: 20,
    paddingVertical: 12
  },
  label: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 6,
  },
  confidence: {
    fontSize: 20,
    color: '#ddd',
  },
  button: {
    position: 'absolute',
    bottom: 30,
    backgroundColor: '#007AFF',
    paddingVertical: 14,
    paddingHorizontal: 32,
    borderRadius: 10,
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
  },
  noDetection: {
    fontSize: 20,
    color: '#fff',
    textAlign: 'center',
  },
  recommendation: {
    fontSize: 18,
    color: 'orange',
    marginTop: 10,
    textAlign: 'center',
  },

});
