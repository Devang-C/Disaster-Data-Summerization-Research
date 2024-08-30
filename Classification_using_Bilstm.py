import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec, KeyedVectors

# Load the pre-trained Word2Vec model
word2vec_model = KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin',binary=True)


# Load the saved LSTM model
model = load_model('models/bilstm_word2vec_model_updated.h5')

dataset_path = 'Data/tweets_data_translated_and_cleaned.csv'
df_turkey = pd.read_csv(dataset_path)
df_turkey = df_turkey.dropna(subset=['translation'])

# df_turkey = df_turkey.head(1000)
# Tokenize and pad the text data
max_words = 10000  
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(df_turkey['translation'])

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(df_turkey['translation'])
padded_sequences = pad_sequences(sequences, maxlen=model.input_shape[1], padding='post')

# Create an embedding matrix using Word2Vec embeddings
embedding_matrix = np.zeros((max_words, word2vec_model.vector_size))
for word, i in tokenizer.word_index.items():
    if i < max_words and word in word2vec_model:
        embedding_matrix[i] = word2vec_model[word]

predictions = model.predict(padded_sequences)
predicted_classes = np.argmax(predictions, axis=1)

# Map predicted class indices to class labels using the dictionary
class_labels_dict = {0: 'caution_and_advice', 1: 'displaced_people_and_evacuations', 2: 'infrastructure_and_utility_damage', 3: 'injured_or_dead_people', 4: 'not_humanitarian', 5: 'other_relevant_information', 6: 'requests_or_urgent_needs', 7: 'rescue_volunteering_or_donation_effort', 8: 'sympathy_and_support'}
df_turkey['predicted_class'] = predicted_classes
df_turkey['predicted_class'] = df_turkey['predicted_class'].map(class_labels_dict)

# Save the classified dataset to a new CSV file
output_csv_path = 'Data/turkey_dataset_classified_bi_lstm.csv'
df_turkey.to_csv(output_csv_path, index=False)
