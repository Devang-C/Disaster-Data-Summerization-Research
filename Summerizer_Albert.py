# ALBERT SUMMARIZATION ON CLASSIFIED DATA

import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from rouge import Rouge
import numpy as np
import statistics
import time
class AlbertTextSummarizer:
    def __init__(self):
        self.model_name = 'paraphrase-albert-small-v2'
        self.model = SentenceTransformer(self.model_name)

    def jensen_shannon_divergence(self, p, q):
        m = (p + q) / 2.0
        epsilon = np.finfo(float).eps
        p = np.clip(p, epsilon, 1 - epsilon)
        q = np.clip(q, epsilon, 1 - epsilon)

        # Check if both p and q are zero vectors
        if np.all((p == 0)) and np.all((q == 0)):
            return 0

        divergence = 0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m)))
        return divergence

    def get_word_probabilities(self, word_list, common_word_set):
        word_count = len(word_list)

        if word_count==0:
            return np.zeros(len(common_word_set))

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

    def calculate_rouge_l_score(self, system_summary, reference_summary):
        rouge = Rouge()
        rouge_scores = rouge.get_scores(system_summary, reference_summary)
        if not rouge_scores:
            return 0.0  # Return 0 if rouge_scores is empty
        rouge_l_score = rouge_scores[0]["rouge-l"]["f"]
        return rouge_l_score

    def extractive_summarization_albert(self, text):
        sentences = [sentence.strip() for sentence in text.split('.') if sentence]
        num_sentences = int(len(sentences) / 2)

        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings)

        top_sentence_indices = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings).argsort(descending=True)

        # print("\nTOP SELECTED INDICES: ",top_sentence_indices)
        selected_sentences = [sentences[i] for i in top_sentence_indices[0][:num_sentences]]
        summary = ' '.join(selected_sentences)

        return summary

    def summarize_text(self, timestamp, document=None, cleaned_text_rows=None, summary_ratio=0.5):
        if document:
            sentences = nltk.sent_tokenize(document)
        elif cleaned_text_rows:
            sentences = [nltk.sent_tokenize(text) for text in cleaned_text_rows]
            sentences = [sentence for sublist in sentences for sentence in sublist]
        else:
            raise ValueError("Either 'document' or 'cleaned_text_rows' must be provided.")

        # Join all sentences to create the input for summarization
        
        input_for_summarization = ' '.join(sentences)

        # Generate summary using ALBERT
        albert_summary = self.extractive_summarization_albert(input_for_summarization)
        if not albert_summary.strip():
            print("\n\nNO SUMMARY GENERATED. THE INPUT TEXT IS: ",input_for_summarization)
            return {}

        common_word_set = set()
        for sentence in [input_for_summarization, albert_summary]:
            words = nltk.word_tokenize(sentence.lower())
            common_word_set.update(words)

        js_divergence = self.calculate_sentence_js_divergence(albert_summary, input_for_summarization, common_word_set)
        rouge_l_score = self.calculate_rouge_l_score(albert_summary, input_for_summarization)

        result = {
            'timestamp': timestamp,
            'js_score': js_divergence,
            'rouge_l_score': rouge_l_score,
            'generated_summary': albert_summary,
            'actual_text': input_for_summarization
        }

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
                top_priorities = [ 'injured_or_dead_people','requests_or_urgent_needs','missing_or_found_people', 'infrastructure_and_utility_damage','rescue_volunteering_or_donation_effort','displaced_people_and_evacuations']
            elif pd.Timestamp('2023-02-08', tz='UTC') < name <= pd.Timestamp('2023-02-18', tz='UTC'):
                top_priorities = ['injured_or_dead_people', 'infrastructure_and_utility_damage', 'rescue_volunteering_or_donation_effort', 'other_relevant_information']
            else:
                top_priorities = ['other_relevant_information', 'sympathy_and_support', 'not_humanitarian', 'rescue_volunteering_or_donation_effort']

            filtered_rows = group[group['predicted_class'].isin(top_priorities)]
            remaining_tweets = filtered_rows.copy()

            while not remaining_tweets.empty:
                max_tweets = min(remaining_tweets.shape[0], np.random.randint(5, 13)) # SELECTING A RANDOM NUMBER OF TWEETS TO SUMMERIZE 
                sampled_rows = remaining_tweets.sample(n=max_tweets, replace=False)
                text = ' '.join(sampled_rows['cleaned_translation_bing'].tolist())
                result = self.summarize_text(timestamp=name, document=text)
                results.append(result)
                remaining_tweets.drop(sampled_rows.index, inplace=True)

        df_results = pd.DataFrame(results)
        df_results.to_csv('Data/summary_results_albert.csv', index=False)

# Example usage
if __name__ == "__main__":
    start_time = time.time()
    albert_summarizer = AlbertTextSummarizer()

    csv_path = "Data/turkey_tweets_classified.csv" 
    albert_summarizer.summarize_cleaned_text_by_time_interval(csv_path)

    end_time = time.time()

    time_required = end_time - start_time

    print(f"Time required for execution(with classification): {time_required} seconds")

