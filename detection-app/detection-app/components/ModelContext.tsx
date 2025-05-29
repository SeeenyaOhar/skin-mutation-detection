// components/TfModelContext.tsx
import React, { createContext, useContext, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';

interface TfModelContextType {
  model: tf.GraphModel | null;
  loading: boolean;
}

const TfModelContext = createContext<TfModelContextType>({
  model: null,
  loading: true,
});

export const useTfModel = () => useContext(TfModelContext);

export const ModelProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [model, setModel] = useState<tf.GraphModel | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.ready();
        await tf.setBackend('rn-webgl');

        const model = await tf.loadGraphModel(
          bundleResourceIO(
            require('../assets/model/model(2).json'),
            require('../assets/model/model(2).bin')
          )
        );

        setModel(model);
      } catch (e) {
        console.log('[LOADING ERROR] info:', e);
      } finally {
        setLoading(false);
      }
    };

    loadModel();
  }, []);

  return (
    <TfModelContext.Provider value={{ model, loading }}>
      {children}
    </TfModelContext.Provider>
  );
};
