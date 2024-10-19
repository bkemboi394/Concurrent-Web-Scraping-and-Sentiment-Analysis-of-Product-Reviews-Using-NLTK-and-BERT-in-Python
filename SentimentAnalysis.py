import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import json
import os


# Class for managing the review scraping process
class ReviewScraper:
    def __init__(self, product, product_id, scraper_api_key):
        self.product = product
        self.product_id = product_id
        self.scraper_api_key = scraper_api_key
        self.base_url = f"https://www.amazon.com/{product}/product-reviews/{product_id}/ref=cm_cr_getr_d_paging_btm_prev_1?ie=UTF8&reviewerType=all_reviews&pageNumber=1"

    def get_soup(self, url):
        # Requesting the page using ScraperAPI
        params = {
            'api_key': self.scraper_api_key,
            'url': url,
            'keep_headers': 'true'
        }
        response = requests.get(f'http://api.scraperapi.com/', params=urlencode(params))

        # Checking if the request was successful
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
        else:
            print(f"Error fetching page: {response.status_code}")
            return None

    # Concurrently scraping each page and returning the results in order
    def scrape_reviews_concurrently(self, total_pages):
        reviews = []
        titles = {}
        product_star_rating = None

        def scrape_page(page_num):
            current_url = f"https://www.amazon.com/{self.product}/product-reviews/{self.product_id}/ref=cm_cr_getr_d_paging_btm_prev_{page_num}?ie=UTF8&reviewerType=all_reviews&pageNumber={page_num}"
            print(f"Scraping page {page_num}...")  # To track progress
            soup = self.get_soup(current_url)
            if not soup:
                return page_num, None, None, None

            if page_num == 1:
                # Scraping the star rating only from the first page
                product_star_rating_element = soup.find("span", {"data-hook": "rating-out-of-text"})
                if product_star_rating_element:
                    star_rating = product_star_rating_element.get_text().strip()
                else:
                    star_rating = "Not found"
            else:
                star_rating = None

            # Scraping review titles and contents
            review_titles = {i + 1: item.get_text().strip().split('\n')[1] for i, item in enumerate(soup.find_all("a", "review-title"))}

            review_contents = [item.get_text().strip() for item in soup.find_all("span", {"data-hook": "review-body"})]

            return page_num, review_titles, review_contents, star_rating

        # Using ThreadPoolExecutor for concurrent scraping
        with ThreadPoolExecutor() as executor:
            future_to_page = {executor.submit(scrape_page, page_num): page_num for page_num in range(1, total_pages + 1)}

            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    page_num, page_titles, page_reviews, star_rating = future.result()

                    # Adding to global reviews and titles, maintaining the order
                    if page_titles and page_reviews:
                        titles.update({i + (page_num - 1) * 10: title for i, title in page_titles.items()})
                        reviews.extend(page_reviews)
                    if star_rating and page_num == 1:
                        product_star_rating = star_rating

                except Exception as exc:
                    print(f"Page {page_num} generated an exception: {exc}")

        return titles, reviews, product_star_rating

    @staticmethod
    def save_reviews_to_file(file_path, titles, reviews, star_rating):
        data = {
            "star_rating": star_rating,
            "titles": titles,
            "reviews": reviews
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Reviews saved to {file_path}")

    @staticmethod
    def load_reviews_from_file(file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            print(f"Loaded reviews from {file_path}")
            return data['titles'], data['reviews'], data['star_rating']
        else:
            print(f"No file found at {file_path}")
            return None, None, None

# SIA sentiment Analysis logic
class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    @staticmethod
    def preprocess_reviews(reviews):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        punctuation_free_reviews = [review.translate(str.maketrans('', '', string.punctuation)) for review in reviews]
        tokenized_reviews = [word_tokenize(review) for review in punctuation_free_reviews]

        # Applying lemmatization in the preprocessing step
        lemmatizer = WordNetLemmatizer()
        cleaned_reviews = [[lemmatizer.lemmatize(token.lower()) for token in tokens if
                            token.isalpha() and token.lower() not in stop_words]
                           for tokens in tokenized_reviews]
        return cleaned_reviews

    @staticmethod
    def load_adjective_vocab():
        # Extract adjectives ('a') from WordNet
        adjectives = set(wordnet.all_lemma_names(pos='a'))
        return list(adjectives)

    @staticmethod
    def validate_reviews(reviews, vocab):
        return [[token for token in review if token in vocab] for review in reviews]

    def analyze_sentiment(self, reviews):
        sentiment_counts = {'very_positive': 0, 'positive': 0, 'negative': 0, 'very_negative': 0, 'neutral': 0}
        review_sentiments = []
        for review in reviews:
            if not review:
                continue
                #Sia sentiment calculation logic
            score = sum(self.sia.polarity_scores(w)["compound"] for w in review) / len(review)
            if score > 0.05:
                sentiment_counts['positive'] += 1
                review_sentiments.append("positive")
            elif score > 0:
                sentiment_counts['very_positive'] += 1
                review_sentiments.append("very positive")
            elif score < -0.05:
                sentiment_counts['negative'] += 1
                review_sentiments.append("negative")
            elif score < 0:
                sentiment_counts['very_negative'] += 1
                review_sentiments.append("very negative")
            else:
                sentiment_counts['neutral'] += 1
                review_sentiments.append("neutral")

        return sentiment_counts, review_sentiments


# Fine-tuning BERT (using Google Colab A100 GPU)
class BERTFineTuner:
    def __init__(self, model_path=None):
        # First checking if a pre-trained model path is provided to load from
        if model_path and os.path.exists(model_path):
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            print(f"Loaded fine-tuned model from {model_path}")
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
            self.dataset = load_dataset('amazon_polarity')
            print("Initialized BERT model from pre-trained weights.")

            # Freezing all BERT layers initially
            for param in self.model.bert.parameters():
                param.requires_grad = False
    
                # Unfreezing the last 5 BERT layers
            bert_layers = self.model.bert.encoder.layer
            for i in range(len(bert_layers)):
    
                if i >= len(bert_layers) - 5:  # Last fivelayers
                    for param in bert_layers[i].parameters():
                        param.requires_grad = True
    
                        # Unfreezing the classifier layer
            for param in self.model.classifier.parameters():
                param.requires_grad = True


    def tokenize_function(self, examples):
        return self.tokenizer(examples['content'], padding="max_length", truncation=True)

    def fine_tune(self, output_dir="./fine_tuned_bert"):
        tokenized_datasets = self.dataset.map(self.tokenize_function, batched=True)
        training_args = TrainingArguments(output_dir="./results", evaluation_strategy="epoch",
                                          per_device_train_batch_size=64, num_train_epochs=3,
                                          logging_steps=50, save_strategy="epoch", fp16=True,
                                          gradient_accumulation_steps=2,
                                          learning_rate=3e-5)

        trainer = Trainer(model=self.model, args=training_args, train_dataset=tokenized_datasets['train'],
                          eval_dataset=tokenized_datasets['test'])
        trainer.train()

        # Save the fine-tuned model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Fine-tuned model saved to {output_dir}")

    def classify_reviews(self, reviews):
        inputs = self.tokenizer(reviews, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        return ["positive" if pred == 1 else "negative" for pred in predictions]





# Results generator for both SIA and BERT
class ReportGenerator:
    @staticmethod
    def display_SIA_results(total_sentiment_counts, product_star_rating, review_sentiments,titles):
        print("SIA Results:\n")
        total_reviews = sum(total_sentiment_counts.values())
        print("\nTotal Reviews =", total_reviews)
        print(
            f"Total Very Positive Reviews = {total_sentiment_counts['very_positive']}\tTotal Positive Reviews = {total_sentiment_counts['positive']}")
        print(
            f"Total Very Negative Reviews = {total_sentiment_counts['very_negative']}\tTotal Negative Reviews = {total_sentiment_counts['negative']}")
        print(f"Total Neutral Reviews = {total_sentiment_counts['neutral']}")

        overall_positive_reviews = "{:.4%}".format(
            (total_sentiment_counts['very_positive'] + total_sentiment_counts['positive']) / total_reviews)
        overall_negative_reviews = "{:.4%}".format(
            (total_sentiment_counts['very_negative'] + total_sentiment_counts['negative']) / total_reviews)
        overall_neutral_reviews = "{:.4%}".format(total_sentiment_counts['neutral'] / total_reviews)

        print("\nOverall Positive Reviews:", overall_positive_reviews)
        print("Overall Negative Reviews:", overall_negative_reviews)
        print("Overall Neutral Reviews:", overall_neutral_reviews)
        print("\nReview sentiments based on SIA:")
        for (idx, sentiment), (idx_title, review_title) in zip(enumerate(review_sentiments, 1), titles.items()):
            print(f"Review {idx}: {review_title} -> {sentiment}")

        overall_positive_reviews = "{:.4%}".format(
            (total_sentiment_counts['very_positive'] + total_sentiment_counts['positive']) / total_reviews)
        print("\nOverall Positive Reviews:", overall_positive_reviews)

        print("\nCompare to the product's star rating of", product_star_rating)


    @staticmethod
    def display_BERT_results(titles, predictions,star_rating):
        positive_reviews = sum(1 for pred in predictions if pred == 'positive')
        total_reviews = len(predictions)
        overall_positive_reviews = "{:.4%}".format(positive_reviews / total_reviews)
        print("\n BERT Results:\n")
        print("\nReview sentiments based on BERT:")
        for idx, title in titles.items():
            print(f"Review {idx}: {title} -> {predictions[idx - 1]}")
        print("Overall positive reviews:", overall_positive_reviews, "\nCompare to the product's star rating of", star_rating)



def main():
    product = input("Enter the product name (e.g., 'Apple-Generation-Cancelling-Transparency-Personalized'): ")
    product_id = input("Enter the product ID (e.g., 'B0CHWRXH8B'): ")
    scraper_api_key = input("Enter your ScraperAPI key: ")

    scraper = ReviewScraper(product, product_id, scraper_api_key)


    file_path = f"{product_id}_reviews.json"

    # First, trying to load reviews from the file
    titles, reviews, product_star_rating = scraper.load_reviews_from_file(file_path)

    # If no file exists, scrape the reviews and save them to the file
    if titles is None or reviews is None:
        titles, reviews, product_star_rating = scraper.scrape_reviews_concurrently(total_pages=10)
        print(product_star_rating)
        scraper.save_reviews_to_file(file_path, titles, reviews, product_star_rating)

    print(f"SENTIMENT ANALYSIS OF {product.upper()} AMAZON'S REVIEWS\n")

    sentiment_analyzer = SentimentAnalyzer()

    # Preprocess the reviews (tokenization, lemmatization, stopword removal)
    cleaned_reviews = sentiment_analyzer.preprocess_reviews(reviews)

    # Load adjective vocabulary for sentiment validation
    adj_vocab = sentiment_analyzer.load_adjective_vocab()

    # Validate reviews against the adjective vocabulary
    validated_reviews = sentiment_analyzer.validate_reviews(cleaned_reviews, adj_vocab)

    # Analyze sentiment of the reviews
    sentiment_counts, review_sentiments = sentiment_analyzer.analyze_sentiment(validated_reviews)

    # Generate the SIA report
    report_generator = ReportGenerator()
    report_generator.display_SIA_results(sentiment_counts, product_star_rating,review_sentiments, titles)

    # Path to save or load the fine-tuned BERT model
    bert_model_path = "./fine_tuned_bert"

    # Check if a fine-tuned model already exists
    if os.path.exists(bert_model_path):
        # Load the fine-tuned BERT model
        fine_tuner = BERTFineTuner(model_path=bert_model_path)
    else:
        # Fine-tune a new BERT model
        fine_tuner = BERTFineTuner()
        fine_tuner.fine_tune(output_dir=bert_model_path)

    # Apply the fine-tuned BERT model to classify sentiment on scraped reviews
    predictions = fine_tuner.classify_reviews(reviews)

    # Generate BERT report
    report_generator = ReportGenerator()
    report_generator.display_BERT_results(titles, predictions,product_star_rating)



if __name__ == "__main__":
    main()
