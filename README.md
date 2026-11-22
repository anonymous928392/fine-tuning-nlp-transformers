# Fine-tuning de Transformers para tres tareas de NLP

Práctica 2 del curso Fundamentos de Procesamiento de Lenguaje Natural, Universidad de Antioquia.
Autor: Brayan Santiago Ramírez Silva.

Se ajustan tres modelos preentrenados de Hugging Face, uno por cada tipo de problema de NLP, siguiendo el mismo flujo: descripción del dataset, análisis exploratorio, búsqueda de hiperparámetros en una muestra, entrenamiento con el dataset completo y evaluación en test.

| Tarea | Dataset | Modelo base | Métrica |
|---|---|---|---|
| Clasificación de secuencias (emociones en tweets) | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) | `distilbert-base-uncased` | F1 macro |
| Etiquetado de secuencias (NER) | CoNLL-2003 | `bert-base-cased` | F1 por entidad (seqeval) |
| Secuencia a secuencia (resumen de diálogos) | [knkarthick/dialogsum](https://huggingface.co/datasets/knkarthick/dialogsum) | `facebook/bart-base` | ROUGE-1/2/L |

## Video de sustentación

https://youtu.be/ebOfBsdwwc8

## Archivos

- `Practica2_FineTuning_NLP.ipynb`: notebook con todo el desarrollo y las salidas.
- `README.md`: este archivo.

## Cómo ejecutarlo

1. Abrir el notebook en Google Colab.
2. Entorno de ejecución > Cambiar tipo de entorno de ejecución > T4 GPU.
3. Ejecutar todas las celdas. En una T4 tarda alrededor de una hora (el entrenamiento final toma 2.4, 6 y 22 minutos por tarea); la primera celda instala `datasets`, `seqeval`, `rouge_score` y `accelerate`.

Los parámetros de la búsqueda (tamaño de las muestras, épocas, grilla de learning rates y batch sizes) están en la celda de configuración al inicio, por si se quiere reducir el tiempo.

## Metodología

- Búsqueda de hiperparámetros: grilla de learning rate {1e-5, 3e-5, 5e-5, 1e-4} por batch size {16, 32} en clasificación y NER, con 3 000 ejemplos y 2 épocas. En resumen, learning rate {2e-5, 5e-5, 1e-4, 2e-4} con batch 8 y 2 000 diálogos, evaluando ROUGE sobre 200 diálogos de validación.
- Entrenamiento final: la mejor configuración con todo el train, evaluación por época en validación y se conserva el mejor checkpoint.
- Optimizador AdamW, weight decay 0.01, warmup del 10 % y decaimiento lineal, precisión mixta fp16.

## Resultados en test

| Tarea | Mejor configuración | Resultado |
|---|---|---|
| Emotion | lr = 1e-4, batch = 16, 3 épocas | F1 macro = 0.892, accuracy = 0.933 |
| CoNLL-2003 | lr = 1e-4, batch = 16, 3 épocas | F1 entidades = 0.907 (P = 0.900, R = 0.914) |
| DialogSum | lr = 2e-4, batch = 8, mejor checkpoint en la época 3 | ROUGE-1 = 51.5, ROUGE-2 = 26.6, ROUGE-L = 43.1 |

- En clasificación y NER el learning rate fue decisivo: en la búsqueda, pasar de 1e-5 a 1e-4 subió el F1 macro de 0.43 a 0.88 y el F1 de entidades de 0.82 a 0.93. Batch 16 superó a batch 32 en todas las comparaciones.
- En resumen, el fine-tuning lleva ROUGE-L de 16.9 (BART sin ajustar) a 43.1; ahí el learning rate casi no cambió el resultado.
- Clases más difíciles: `surprise` (F1 0.768) y `love` (0.830) en emociones, `MISC` (0.796) en NER.

El detalle de cada búsqueda, las curvas de entrenamiento, los reportes por clase, las matrices de confusión y ejemplos de predicción están en el notebook.

## Modelos publicados

- https://huggingface.co/anonymous-666/distilbert-emotion
- https://huggingface.co/anonymous-666/bert-base-cased-conll2003-ner
- https://huggingface.co/anonymous-666/bart-base-dialogsum

## Decisiones

- CoNLL-2003 se carga desde una copia en parquet porque `datasets` 4.x ya no ejecuta scripts de carga.
- En NER la etiqueta de cada palabra se asigna a su primer subtoken; el resto se marca con -100 para que la pérdida lo ignore.
- En DialogSum cada diálogo de test tiene tres resúmenes de referencia; ROUGE se calcula contra la mejor de las tres.
- Se usa BART y no T5 para el resumen porque entrena estable en fp16 en la GPU de Colab.
- Limitaciones: los tres datasets están en inglés y ROUGE no mide si el resumen es fiel al diálogo.

## Referencias

- Hugging Face LLM Course: https://huggingface.co/learn/llm-course
- Saravia et al. (2018). CARER: Contextualized Affect Representations for Emotion Recognition. EMNLP.
- Tjong Kim Sang y De Meulder (2003). Introduction to the CoNLL-2003 Shared Task. CoNLL.
- Chen et al. (2021). DialogSum: A Real-Life Scenario Dialogue Summarization Dataset. Findings of ACL.
- Sanh et al. (2019). DistilBERT. Devlin et al. (2019). BERT. Lewis et al. (2020). BART.
