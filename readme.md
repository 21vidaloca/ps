Aceasta este documentația tehnică și de utilizare pentru scriptul
`main.py`, care implementează un clasificator de spam folosind
algoritmul Multinomial Naive Bayes.

# Modelul Matematic (Multinomial Naive Bayes)

Modelul implementat este **Multinomial Naive Bayes**, un algoritm
probabilistic popular pentru clasificarea textului. Scopul este de a
determina cea mai probabilă clasă (de exemplu, `spam` sau `ham`) pentru
un document (mesaj) dat.

Acesta se bazează pe **Teorema lui Bayes**:

$$P(C|D) = \frac{P(D|C) \cdot P(C)}{P(D)}$$

Unde:

-   $C$ este o clasă (ex: `spam`).

-   $D$ este documentul (mesajul).

-   $P(C|D)$ este probabilitatea (posterioară) ca documentul să aparțină
    clasei $C$, știind conținutul documentului.

-   $P(D|C)$ este probabilitatea (verosimilitatea) de a observa
    documentul $D$, știind că aparține clasei $C$.

-   $P(C)$ este probabilitatea (apriori) a clasei $C$.

-   $P(D)$ este probabilitatea documentului $D$.

Pentru clasificare, dorim să găsim clasa $C$ care maximizează această
probabilitate. Deoarece $P(D)$ este constant pentru toate clasele, îl
putem ignora:

$$C_{\text{final}} = \text{max}_{c \in \{\text{spam}, \text{ham}\}} P(D|C) \cdot P(C)$$

**Asumpția \"Naivă\":** Algoritmul presupune că toate cuvintele
(token-urile) $w_i$ din document sunt **independente condiționat** de
clasă.

$$P(D|C) = P(w_1, w_2, \dots, w_n | C) \approx \prod_{i=1}^{n} P(w_i | C)$$

Combinând, formula devine:

$$C_{\text{final}} = \text{max}_{c} P(C) \cdot \prod_{i=1}^{n} P(w_i | C)$$

## Implementarea în Cod

Deoarece înmulțirea unui număr mare de probabilități mici (între 0 și 1)
poate duce la *aritmetic underflow* (rezultate care tind spre zero),
scriptul lucrează cu **logaritmii probabilităților**. Transformarea este
monotonă, deci clasa care maximizează produsul va maximiza și suma
logaritmică:

$$C_{\text{final}} = \text{max}_{c} \left( \log(P(C)) + \sum_{i=1}^{n} \log(P(w_i | C)) \right)$$

Acest lucru este reflectat direct în funcția `predict`:
* `score = self.log_prior[c]` (termenul $\log(P(C))$ )
* `score += self.log_likelihood[c][word]` (termenul $\sum \log(P(w_i | C))$ )

## Calculul Probabilităților

### Probabilitatea Prior (A Priori) $P(C)$

Aceasta este probabilitatea ca un document ales la întâmplare să
aparțină clasei $C$. Se calculează ca fracțiunea documentelor de
antrenare care aparțin acelei clase.

-   **Formula:**
    $P(C) = \frac{\text{Număr documente în clasa } C}{\text{Număr total documente}}$

-   **Cod:**
    `self.log_prior[c] = math.log(num_docs_per_class[c] / num_messages)`

### Probabilitatea Likelihood (Verosimilitatea) $P(w|C)$

Aceasta este probabilitatea ca un cuvânt $w$ să apară, dat fiind că
documentul aparține clasei $C$.

-   **Formula (Multinomială):**
    $P(w|C) = \frac{\text{Număr apariții } w \text{ în clasa } C}{\text{Număr total cuvinte în clasa } C}$

**Netezirea (Smoothing):** O problemă apare dacă un cuvânt din setul de
testare nu a fost văzut niciodată în setul de antrenare pentru o anumită
clasă. Acest lucru ar duce la o probabilitate de 0, care, prin înmulțire
sau $\log(0)$ ar anula scorul întregii clase.

Pentru a preveni asta, se folosește **Netezirea Laplace (sau Add-k)**.
Scriptul folosește `alpha=1.0` (Add-1).

-   **Formula cu Netezire Laplace:**
    $$P(w|C) = \frac{\text{Număr apariții } w \text{ în clasa } C + \alpha}{\text{Număr total cuvinte în clasa } C + \alpha \cdot V}$$

-   Unde:

    -   $\alpha$ este parametrul de netezire (1.0 în cod).

    -   $V$ este dimensiunea vocabularului (numărul total de cuvinte
        unice).

-   **Cod:**

    -   `numerator = count + self.alpha`

    -   `denominator = total_words_per_class[c] + self.alpha * V`

    -   `self.log_likelihood[c][word] = math.log(numerator / denominator)`

**Cuvinte Necunoscute:** Pentru cuvintele întâlnite la predicție, dar
care nu există în vocabularul de antrenare, se folosește o probabilitate
specială (cu $\text{Număr apariții } w = 0$):

-   **Cod:**
    `self.log_likelihood[c][’_UNKNOWN_’] = math.log(self.alpha / denominator)`

# Structura Codului și Funcții Principale

Scriptul este compus dintr-o singură clasă, `MultinomialNaiveBayes`, și
un bloc de execuție principal care gestionează datele și rulează
modelul.

## Importuri și Configurare

-   `re`: Pentru expresii regulate (folosit la curățarea textului).

-   `math`: Pentru funcția `math.log`.

-   `random`: Pentru amestecarea datelor (`random.shuffle`).

-   `csv`: Pentru citirea fișierului de date.

-   `sys` și `defaultdict`: Utilitare standard.

-   **Blocul `csv.field_size_limit`:** Acesta este un fix de robustețe.
    Unele mesaje (email-uri) pot fi foarte lungi, depășind limita
    implicită a cititorului CSV. Acest bloc încearcă să seteze limita la
    cea mai mare valoare posibilă pentru a preveni erorile de tip
    `Error: field larger than field limit`.

## Clasa `MultinomialNaiveBayes`

-   **`__init__(self, alpha=1.0)`**

    -   Constructorul clasei.

    -   `self.alpha`: Stochează parametrul de netezire Laplace.

    -   `self.log_prior`: Dicționar pentru a stoca $\log(P(C))$.

    -   `self.log_likelihood`: Dicționar de dicționare pentru a stoca
        $\log(P(w|C))$.

    -   `self.vocab`: Un set care va conține toate cuvintele unice din
        datele de antrenare.

    -   `self.classes`: Un set care va conține etichetele unice (ex:
        'spam', 'ham').

-   **`_tokenize(self, text)`**

    -   Funcție internă de pre-procesare a textului.

    -   Transformă textul în minuscule (`text.lower()`).

    -   Elimină toate caracterele care nu sunt litere sau spații
        (`re.sub(r’[^a-z\s]’, ”, ...)`).

    -   Împarte textul într-o listă de cuvinte (tokeni).

-   **`fit(self, X_train, y_train)`**

    -   Funcția de **antrenare** a modelului.

    -   Primește listele de mesaje (`X_train`) și etichetele
        corespunzătoare (`y_train`).

    -   **Pași:**

        1.  Iterează prin toate mesajele și etichetele.

        2.  Calculează `num_docs_per_class` (câte documente sunt în
            fiecare clasă).

        3.  Tokenizează fiecare mesaj.

        4.  Construiește vocabularul (`self.vocab`).

        5.  Numără aparițiile fiecărui cuvânt în fiecare clasă
            (`word_counts_per_class`).

        6.  Calculează numărul total de cuvinte per clasă
            (`total_words_per_class`).

        7.  Calculează `self.log_prior` pentru fiecare clasă (conform
            formulei de Prior).

        8.  Calculează `self.log_likelihood` pentru fiecare cuvânt
            (conform formulei cu netezire).

        9.  Calculează o probabilitate specială `_UNKNOWN_` pentru
            cuvintele nevăzute.

-   **`predict(self, X_test)`**

    -   Funcția de **predicție**.

    -   Primește o listă de mesaje noi (`X_test`) pentru a le clasifica.

    -   **Pași:**

        1.  Iterează prin fiecare mesaj `x` de testat.

        2.  Tokenizează mesajul.

        3.  Calculează un scor pentru fiecare clasă $c$:

            -   Începe cu `score = self.log_prior[c]`.

            -   Adaugă `self.log_likelihood[c][word]` pentru fiecare
                cuvânt.

            -   Dacă un cuvânt nu este în `self.vocab`, adaugă
                `self.log_likelihood[c][’_UNKNOWN_’]`.

        4.  Selectează clasa (`predicted_class`) cu cel mai mare scor
            logaritmic.

        5.  Returnează o listă cu toate predicțiile.

## Blocul Principal de Execuție

Această parte a scriptului încarcă datele și orchestrează procesul de
antrenare și testare.

1.  **Încărcarea Datelor:** Deschide `combined_data.csv`, citește datele
    (sărind peste antet) și populează listele `messages` și `labels`.

2.  **Pregătirea Datelor:** Combină, amestecă (folosind
    `random.seed(42)` pentru reproductibilitate) și împarte datele în
    80% antrenare și 20% testare.

3.  **Antrenarea Modelului:** Creează o instanță `MultinomialNaiveBayes`
    și apelează `model.fit()`.

4.  **Testarea și Evaluarea:** Apelează `model.predict()` pe setul de
    test și calculează acuratețea.

5.  **Exemple Noi:** Definește și testează două mesaje noi (`test_spam`,
    `test_ham`) și afișează predicția.

# Instrucțiuni de Utilizare

## Cerințe

-   Python 3.x

-   Un fișier numit `combined_data.csv` în același director cu
    `main.py`.

## Formatul Datelor (`combined_data.csv`)

Scriptul așteaptă un fișier CSV cu un rând de antet (ignorat) și
următoarele coloane:

-   **Coloana 1 (index 0):** Eticheta clasei (ex: `spam` sau `ham`).

-   **Coloana 2 (index 1):** Textul mesajului.

**Exemplu `combined_data.csv`:**

    label,text
    ham,"Hey, are you around? I'm running a bit late..."
    spam,"Congratulations! You've won a free iPhone. Click here..."
    ham,"Ok, see you in 5."
    ...

## Rulare

1.  Asigurați-vă că fișierul `combined_data.csv` este în directorul
    corect.

2.  Deschideți un terminal sau o linie de comandă.

3.  Navigați în directorul care conține `main.py`.

4.  Rulați scriptul folosind comanda:

```{=html}
<!-- -->
```
    python main.py

# Exemple de Utilizare

## Outputul Așteptat la Rulare

La rularea scriptului, veți vedea un output similar cu următorul:

    Am incarcat 83448 mesaje.
    Impartire date: 66758 antrenare, 16690 testare.
    Se antreneaza modelul Naive Bayes (cu filtrare stop-words)...
    Antrenare finalizata.

    --- Rezultate ---
    Acuratete pe setul de test: 97.45%

    --- Testeaza un mesaj nou ---
    Test Spam: 'Hi im Ana with a free offer meal' -> Predictie: ['spam']
    Test Ham:  'Hey, are you around? I'm running a bit late with the lunch.' -> Predictie: ['ham']

## Testarea unor Mesaje Noi

Pentru a testa propriile mesaje, puteți modifica variabilele `test_spam`
și `test_ham` direct în fișierul `main.py` înainte de a-l rula:

    # Modificați aceste linii la sfârșitul fișierului main.py

    print("\n--- Testeaza un mesaj nou ---")

    test_spam = "URGENT! Your account is compromised. Click link to verify!"
    test_ham = "Can you please send me the report from yesterday's meeting?"

    print(f"Test 1: '{test_spam}' -> Predictie: {model.predict([test_spam])[0]}")
    print(f"Test 2: '{test_ham}' -> Predictie: {model.predict([test_ham])[0]}")

# Referințe Bibliografice

1.  **Setul de Date (Sursă):**\
    Singh, P. (2023). *Email Spam Classification Dataset*. Kaggle.
    Disponibil la:
    <https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset>

2.  **Implementare de Referință (pentru studiu comparativ):**\
    Scikit-learn. (n.d.). *Naive Bayes*.
    `sklearn.naive_bayes.MultinomialNB`. Disponibil la:
    <https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes>
