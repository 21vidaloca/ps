import re
import math
import random
import csv
import sys
from collections import defaultdict
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)
# --- Sfârșitul blocului nou ---
class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0): # netezire Laplace
        self.alpha = alpha
        
        self.log_prior = {} # log(P(ham)) sa log(P(spam))
        self.log_likelihood = {} # log(P('free' | spam))
        
        self.vocab = set() # cuvinte unice
        
        self.classes = set() # etichete unice spam sau ham
        
        # self.stop_words = set([
        #     'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an',
        #     'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been',
        #     'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cannot',
        #     'could', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does',
        #     'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few',
        #     'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't",
        #     'have', 'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself',
        #     'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't",
        #     'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'might', 'more',
        #     'most', 'must', 'my', 'myself', 'no', 'nor', 'not', 'now', 'o', 'of', 'off',
        #     'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out',
        #     'over', 'own', 're', 's', 'same', 'she', "she's", 'should', "should've",
        #     'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll",
        #     'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these',
        #     'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
        #     've', 'very', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't",
        #     'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will',
        #     'with', 'won', "won't", 'would', 'y', 'you', "you'd", "you'll", "you're",
        #     "you've", 'your', 'yours', 'yourself', 'yourselves'
        # ])

    def _tokenize(self, text):
        text_curat = re.sub(r'[^a-z\s]', '', text.lower())
        all_words = text_curat.split()
        return [word for word in all_words]# if word not in self.stop_words]

    def fit(self, X_train, y_train):
        num_messages = len(X_train)
        num_docs_per_class = defaultdict(int)
        word_counts_per_class = defaultdict(lambda: defaultdict(int))
        total_words_per_class = defaultdict(int)
        
        self.classes = set(y_train)

        for x, y in zip(X_train, y_train):
            num_docs_per_class[y] += 1
            
            words = self._tokenize(x) 
            
            for word in words:
                self.vocab.add(word)
                word_counts_per_class[y][word] += 1
                total_words_per_class[y] += 1

        for c in self.classes:
            self.log_prior[c] = math.log(num_docs_per_class[c] / num_messages)
        
        V = len(self.vocab)
        
        for c in self.classes:
            self.log_likelihood[c] = {}
            
            denominator = total_words_per_class[c] + self.alpha * V
            
            for word in self.vocab:
                count = word_counts_per_class[c].get(word, 0)
                numerator = count + self.alpha
                self.log_likelihood[c][word] = math.log(numerator / denominator)
                
            self.log_likelihood[c]['_UNKNOWN_'] = math.log(self.alpha / denominator)

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            words = self._tokenize(x)
            scores = {}
            
            for c in self.classes:
                score = self.log_prior[c]
                for word in words:
                    if word in self.vocab:
                         score += self.log_likelihood[c][word]
                    else:
                         score += self.log_likelihood[c]['_UNKNOWN_']
                
                scores[c] = score
            
            predicted_class = max(scores, key=scores.get)
            predictions.append(predicted_class)
            
        return predictions

messages = []
labels = []
with open('combined_data.csv', mode='r', encoding='latin-1') as file:
    reader = csv.reader(file)
    header = next(reader) 
    for row in reader:
        if len(row) >= 2:
            labels.append(row[0])
            messages.append(row[1])
                

print(f"Am incarcat {len(messages)} mesaje.")

combined_data = list(zip(messages, labels))
random.seed(42) # Pentru rezultate reproductibile
random.shuffle(combined_data)

split_index = int(len(combined_data) * 0.80)
train_data = combined_data[:split_index]
test_data = combined_data[split_index:]

X_train = [msg for msg, label in train_data]
y_train = [label for msg, label in train_data]
X_test = [msg for msg, label in test_data]
y_test = [label for msg, label in test_data]

print(f"Impartire date: {len(X_train)} antrenare, {len(X_test)} testare.")

# 3. Antrenarea Modelului
print("Se antreneaza modelul Naive Bayes (cu filtrare stop-words)...")
model = MultinomialNaiveBayes(alpha=1.0)
model.fit(X_train, y_train)
print("Antrenare finalizata.")

# 4. Testarea Modelului
predictions = model.predict(X_test)

correct_predictions = 0
for pred, actual in zip(predictions, y_test):
    if pred == actual:
        correct_predictions += 1
        
accuracy = (correct_predictions / len(y_test)) * 100
print(f"\n--- Rezultate ---")
print(f"Acuratete pe setul de test: {accuracy:.2f}%")

print("\n--- Testeaza un mesaj nou ---")
# 1 e spam
# 0 e ham
test_spam = "Hi im Ana with a free offer meal"
test_ham = "Hey, are you around? I'm running a bit late with the lunch."

print(f"Test Spam: '{test_spam}' -> Predictie: {model.predict([test_spam])[0]}")
print(f"Test Ham:  '{test_ham}' -> Predictie: {model.predict([test_ham])[0]}")