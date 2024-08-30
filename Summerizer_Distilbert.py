from transformers import DistilBertModel, DistilBertTokenizer
from summarizer import Summarizer
from rouge import Rouge
import numpy as np
import nltk
import pandas as pd
import sys
import time

class DistilBERTSummarizer:
    def __init__(self, random_state=7):
        self.model_name = 'distilbert-base-uncased'
        self.model = DistilBertModel.from_pretrained(self.model_name, output_hidden_states=True)
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.summarizer = Summarizer(custom_model=self.model, custom_tokenizer=self.tokenizer, random_state=random_state)
        self.rouge = Rouge()

    def jensen_shannon_divergence(self, p, q):
        m = (p + q) / 2.0
        epsilon = np.finfo(float).eps
        p = np.clip(p, epsilon, 1 - epsilon)
        q = np.clip(q, epsilon, 1 - epsilon)
        divergence = 0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m)))
        return divergence

    def get_word_probabilities(self, word_list, common_word_set):
        word_count = len(word_list)
        word_frequencies = {word: word_list.count(word) / word_count for word in common_word_set}
        probability_vector = np.array([word_frequencies[word] for word in common_word_set])
        return probability_vector

    def calculate_sentence_js_divergence(self, sentence1, sentence2, common_word_set):
        words1 = nltk.word_tokenize(sentence1.lower())
        words2 = nltk.word_tokenize(sentence2.lower())
        p = self.get_word_probabilities(words1, common_word_set)
        q = self.get_word_probabilities(words2, common_word_set)
        js_divergence = self.jensen_shannon_divergence(p, q)
        return js_divergence

    def summarize_text(self, timestamp, document=None, cleaned_text_rows=None, summary_ratio=0.5):
        if document:
            sentences = nltk.sent_tokenize(document)
        elif cleaned_text_rows:
            sentences = [nltk.sent_tokenize(text) for text in cleaned_text_rows]
            sentences = [sentence for sublist in sentences for sentence in sublist]
        else:
            raise ValueError("Either 'document' or 'cleaned_text_rows' must be provided.")

        number_sentence = max(min(len(sentences), 8), 20)  # Limit the number of sentences
        input_for_summarization = ' '.join(sentences[:number_sentence])

        print("Input for summarization: ", input_for_summarization)

        # Generate summary using DistilBERT
        distilbert_summary = self.summarizer(input_for_summarization)

        if not distilbert_summary.strip():
            print("NO SUMMARY GENERATED")
            return {}

        # Calculate Rouge-L score
        reference_summary = ' '.join(sentences)
        rouge_scores = self.rouge.get_scores(distilbert_summary, reference_summary)
        rouge_l_score = rouge_scores[0]["rouge-l"]["f"]

        # Calculate JS divergence
        common_word_set = set(nltk.word_tokenize(reference_summary.lower()))
        js_divergence = self.calculate_sentence_js_divergence(distilbert_summary, reference_summary, common_word_set)

        result = {
            'timestamp': timestamp,
            'js_score': js_divergence,
            'rouge_l_score': rouge_l_score,
            'generated_summary': distilbert_summary,
            'actual_text': reference_summary
        }

        print("\n\n summary: ", distilbert_summary)
        print("\n js score: ", js_divergence)
        print("\n rouge-l: ", rouge_l_score)

        return result

    def summarize_cleaned_text_by_time_interval(self, csv_path, batch_size=50):
        df = pd.read_csv(csv_path, parse_dates=['date'], infer_datetime_format=True, encoding='ISO-8859-1')
        # df = df.head(500)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        g = df.groupby(pd.Grouper(freq='D'))

        results = []

        for name, group in g:
            name = name.tz_convert('UTC')

            if name <= pd.Timestamp('2023-02-08', tz='UTC'):
                top_priorities = ['injured_or_dead_people', 'requests_or_urgent_needs', 'missing_or_found_people', 'infrastructure_and_utility_damage', 'rescue_volunteering_or_donation_effort', 'displaced_people_and_evacuations']
            elif pd.Timestamp('2023-02-08', tz='UTC') < name <= pd.Timestamp('2023-02-18', tz='UTC'):
                top_priorities = ['injured_or_dead_people', 'infrastructure_and_utility_damage', 'rescue_volunteering_or_donation_effort', 'other_relevant_information']
            else:
                top_priorities = ['other_relevant_information', 'sympathy_and_support', 'not_humanitarian', 'rescue_volunteering_or_donation_effort']

            filtered_rows = group[group['predicted_class'].isin(top_priorities)]
            remaining_tweets = filtered_rows.copy()

            while not remaining_tweets.empty:
                max_tweets = min(remaining_tweets.shape[0], np.random.randint(5, 13))
                sampled_rows = remaining_tweets.sample(n=max_tweets, replace=False)
                text = ' '.join(sampled_rows['cleaned_translation_bing'].tolist())
                result = self.summarize_text(timestamp=name, document=text)
                results.append(result)
                remaining_tweets.drop(sampled_rows.index, inplace=True)

        df_results = pd.DataFrame(results)
        df_results.to_csv('Data/summary_results_distilbert.csv', index=False)

if __name__ == "__main__":
    start_time = time.time()
    summarizer = DistilBERTSummarizer()

    csv_path = "Data/turkey_dataset_classified_bi_lstm.csv"
    summarizer.summarize_cleaned_text_by_time_interval(csv_path)

    end_time = time.time()

    time_required = end_time - start_time

    print(f"Time required for execution (with classification): {time_required} seconds")