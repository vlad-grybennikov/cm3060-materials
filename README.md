# CM3060 Natural Language Processing – Complete Exam Preparation Guide
*Condensed textbook for University of London BSc Computer Science*

> **Exam format:** Part A – 10 MCQs (40 marks) · Part B – answer 2 of 3 essay/calculation questions (60 marks) · 4 hours · One A4 page of notes permitted · No calculator (show all workings)

---

## Table of Contents
1. [History and Overview of NLP](#1-history-and-overview-of-nlp)
2. [Text Processing Fundamentals](#2-text-processing-fundamentals)
3. [Morphological Processing](#3-morphological-processing)
4. [Corpora and Data](#4-corpora-and-data)
5. [Information Retrieval](#5-information-retrieval)
6. [Language Models and N-grams](#6-language-models-and-n-grams)
7. [Part-of-Speech Tagging](#7-part-of-speech-tagging)
8. [Formal Grammars and Parsing](#8-formal-grammars-and-parsing)
9. [Named Entity Recognition](#9-named-entity-recognition)
10. [Word Representations](#10-word-representations)
11. [Text Classification](#11-text-classification)
12. [Sentiment Analysis](#12-sentiment-analysis)
13. [Information Extraction](#13-information-extraction)
14. [Statistical NLP Evaluation](#14-statistical-nlp-evaluation)
15. [NLP with Python (NLTK and spaCy)](#15-nlp-with-python-nltk-and-spacy)
16. [Regular Expressions](#16-regular-expressions)
17. [Exam Strategy](#17-exam-strategy)
18. [Most Common Exam Questions](#18-most-common-exam-questions)
19. [One-Page Cheat Sheet](#19-one-page-cheat-sheet)

---

## 1. History and Overview of NLP

### 1.1 What is NLP?

Natural Language Processing (NLP) is the computational study of human language. It sits at the intersection of linguistics, computer science, and artificial intelligence. The core challenge: natural language was not designed for machines — it is ambiguous, context-dependent, creative, and constantly evolving.

### 1.2 Three Eras of NLP

| Era | Approximate Dates | Approach | Key Ideas |
|-----|-------------------|----------|-----------|
| **Symbolic / Rule-Based** | 1950s–1980s | Hand-crafted rules, logic | Regular grammars, CFGs, ELIZA (1966), SHRDLU (1972) |
| **Statistical** | 1990s–2010s | Probabilistic models trained on corpora | HMMs, Naive Bayes, n-gram LMs, SVMs |
| **Neural / Deep Learning** | 2010s–present | Representation learning end-to-end | Word2Vec (2013), LSTM, Transformers (2017), BERT (2018), GPT series |

**Key milestones:**
- 1950 — Turing Test proposed
- 1954 — Georgetown–IBM MT experiment
- 1966 — ELIZA (Weizenbaum): pattern-matching chatbot
- 1972 — SHRDLU (Winograd): blocks world NLU
- 1988 — IBM Candide: first statistical MT
- 1993 — Penn Treebank released
- 2003 — Neural language model (Bengio et al.)
- 2013 — Word2Vec (Mikolov et al.)
- 2017 — Attention is All You Need (Vaswani et al.) — Transformer
- 2018 — BERT (Devlin et al.)
- 2022 — ChatGPT (GPT-4 based)

### 1.3 Why is NLP Hard?

**Ambiguity** is the central problem. It manifests at multiple levels:

1. **Lexical ambiguity (polysemy):** *"I went to the bank"* — financial institution or river bank?
2. **Syntactic ambiguity (structural):** *"I saw the man on the hill with a telescope"* — who has the telescope?
3. **Semantic ambiguity:** *"Every student read a book"* — same book or different books?
4. **Pragmatic / discourse ambiguity:** *"Can you pass the salt?"* — request, not a yes/no question.
5. **Referential ambiguity (coreference):** *"John told Peter that his friends were coming"* — whose friends?

Other challenges:
- **Synonymy:** one concept, many words (*smart / bright / clever*)
- **Language is generative:** infinite sentences from finite rules
- **Context dependence:** meaning depends on who says what to whom
- **Negation and coordination:** *"This is not a talk about algorithms"* can confuse a naive classifier
- **Sarcasm and irony:** *"That was a wicked lecture"*
- **Variation:** dialects, registers, slang, code-switching

### 1.4 Rule-Based vs. Statistical NLP

| Feature | Rule-Based | Statistical |
|---------|-----------|-------------|
| Knowledge source | Linguistic expertise | Annotated data |
| Transparency | High | Low (black box) |
| Portability | Low (language-specific rules) | Higher (retrain on new data) |
| Handling of novel input | Poor | Better generalisation |
| Development cost | High (expert time) | High (annotation cost) |
| Example systems | ELIZA, SHRDLU, early parsers | HMM taggers, Naive Bayes classifiers |

**Exam tip:** "Explain differences between rule-based and statistical approaches" — appears frequently. Use this table and give one example of each.

---

## 2. Text Processing Fundamentals

### 2.1 The NLP Pipeline (ASCII Diagram)

```
Raw Text
    │
    ▼
Sentence Tokenisation  ──►  ["Stars are distant suns.", "They glow."]
    │
    ▼
Word Tokenisation       ──►  ["Stars", "are", "distant", "suns", "."]
    │
    ▼
Normalisation           ──►  case folding, punctuation removal
    │
    ▼
Stop Word Removal       ──►  ["Stars", "distant", "suns"]
    │
    ▼
Stemming / Lemmatisation──►  ["star", "distant", "sun"]
    │
    ▼
POS Tagging             ──►  [("Stars","NNS"), ("are","VBP"), ...]
    │
    ▼
Feature Extraction / Indexing
```

**Ordering rationale:** Sentence tokenisation must precede word tokenisation (you need sentences before words). POS tagging requires word tokens. Lemmatisation benefits from POS tags (to determine the correct lemma). Stop word removal and stemming can happen after basic tokenisation. However, for IR tasks, stop word removal often happens before stemming to save computation.

### 2.2 Tokenisation

**Definition:** Tokenisation is the process of segmenting a character sequence into a list of tokens, where a token is an instance of a sequence of characters that form a meaningful unit.

**Challenges (exam-favourite — know at least 4):**

| Challenge | Example | Issue |
|-----------|---------|-------|
| Contractions | *can't* | → *can* + *not*? or *ca* + *n't*? |
| Possessives | *students'*, *student's* | Apostrophe placement |
| Hyphenated words | *long-term*, *state-of-the-art* | Split or keep? |
| URLs | *www.amazon.com* | Don't split on dots |
| Email addresses | *info@abc.com* | Don't split on @ |
| Abbreviations | *Prof.*, *Dr.*, *Ms.* | Dot not a sentence boundary |
| Hashtags | *#NLProc* | Keep as one token |
| Numbers | *1,234,567* or *3.14* | Commas/dots as separators |
| Multi-word expressions | *New York*, *kick the bucket* | Should be one token? |
| Emojis/slang | 😂, *gonna*, *u* | Non-standard tokens |
| German compounding | *Donaudampfschiffahrtsgesellschaft* | One word = multiple concepts |

**NLTK Python example:**
```python
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize

text = "Prof. Russell-Rose can't believe that's the students' long-term prospects."

# Sentence tokenisation
sentences = sent_tokenize(text)
# Result: ["Prof. Russell-Rose can't believe that's the students' long-term prospects."]
# Note: correctly handles "Prof." as abbreviation

# Word tokenisation
tokens = word_tokenize(text)
# Result: ['Prof.', 'Russell-Rose', 'ca', "n't", 'believe', "that's", 
#           'the', 'students', "'", 'long-term', 'prospects', '.']
```

### 2.3 Sentence Tokenisation (Sentence Segmentation)

**Challenge:** The period character '.' signals sentence boundaries but is also used in:
- Abbreviations: *Mr.*, *U.S.A.*, *Prof.*, *TF.IDF*
- Decimal numbers: *3.14*
- URLs: *boohoo.com*
- Ellipsis: *...*
- Emoticons: *:)*

**Approaches:**
1. **Rule-based:** periods followed by capital letters are boundaries; known abbreviation lists
2. **ML-based:** classify each candidate boundary point using features (word before/after, capitalisation, word length)

**Exam example:** *"Ms. Smith likes boohoo.com (but not a lot :)"* — challenges include:
- *Ms.* is an abbreviation, not a sentence end
- *boohoo.com* contains a period that is not a sentence boundary
- The emoticon *:)* contains a closing parenthesis

### 2.4 Types, Tokens, and Terms

**Token:** An individual occurrence of a word in a text. Every instance counts.

**Type:** A unique word form. The vocabulary consists of types.

**Term:** A normalised type used in an IR index (may differ from the raw token, e.g., after stemming).

**Example:** *"The sands of time were eroded by the river of constant change"*

| Count | Value |
|-------|-------|
| Tokens | 13 |
| Types | 11 (note: *the* appears twice, *of* appears twice) |

```
Tokens: The sands of time were eroded by the river of constant change
Count:   1    2   3   4    5     6    7   8   9   10   11     12
                                                       (+the=2nd, of=2nd)
Total tokens = 12 (excluding punctuation if any)
Types: {the, sands, of, time, were, eroded, by, river, constant, change} = 10 unique (after case folding "The"→"the")
```

**Type-token ratio (TTR):** TTR = Types / Tokens. Higher TTR → more lexical diversity.

### 2.5 Normalisation

**Text normalisation** converts text to a canonical form:
- **Case folding:** convert to lowercase (*Apple* → *apple*)
- **Punctuation removal**
- **Number normalisation:** *2024* → `<YEAR>` or keep as-is
- **Expanding contractions:** *can't* → *cannot*
- **Unicode normalisation:** handling accented characters

**Stop words:** High-frequency function words (*the, a, of, in, is*) that carry little discriminative information. Removing them:
- **Advantages:** reduces index size, speeds up retrieval
- **Disadvantages:** loses meaning in phrases (*"To be or not to be"*), may harm queries where stop words are significant

### 2.6 Zipf's Law ⭐ (High-yield MCQ topic)

**Statement:** In any natural language corpus, the frequency of a word is inversely proportional to its rank in the frequency table.

**Formula:**

$$f(w) \propto \frac{1}{r(w)^s}$$

Where:
- $f(w)$ = frequency of word $w$
- $r(w)$ = rank of word $w$ (rank 1 = most frequent)
- $s \approx 1$ for natural language (the exponent, close to 1)

**Equivalently:** $f \times r \approx C$ (a constant)

So the most frequent word appears twice as often as the 2nd most frequent, three times as often as the 3rd, etc.

**Properties (MCQ answers):**
- Word rank and word frequency are **inversely related** ✓
- It describes a **power law** relationship ✓
- It applies to **many naturalistic phenomena** (city populations, earthquake magnitudes, website traffic) ✓
- Word rank and word frequency are *NOT* positively correlated ✗

**Implications for NLP:**
1. A small number of words (*the, of, a, in*) account for a very large fraction of all tokens — these are stop words
2. The vast majority of word types appear very rarely (long tail) — causes **data sparsity** in language models
3. Most words in a test corpus will not have appeared in training — requires **smoothing**

**Heaps' Law (related):** The vocabulary size *V* grows as a function of corpus size *N*: $V \approx kN^\beta$ where $\beta \approx 0.5$. Vocabulary grows, but at a decreasing rate.

**Python: Plotting Zipf's law with NLTK**
```python
import nltk
from nltk.corpus import brown
import matplotlib.pyplot as plt

nltk.download('brown')
fd = nltk.FreqDist(w.lower() for w in brown.words())

# Get frequencies sorted by rank
ranks = range(1, 51)
freqs = [freq for _, freq in fd.most_common(50)]

plt.loglog(list(ranks), freqs)
plt.xlabel("Rank (log scale)")
plt.ylabel("Frequency (log scale)")
plt.title("Zipf's Law — Brown Corpus")
# A straight line on a log-log plot confirms Zipf's law
```

---

## 3. Morphological Processing

### 3.1 Morphemes and Morphology

**Morphology** is the study of the internal structure of words.

A **morpheme** is the smallest meaningful unit of language. It cannot be broken down further without losing meaning.

- **Free morpheme:** can stand alone (*cat*, *run*, *fast*)
- **Bound morpheme:** must attach to another morpheme (*-ing*, *un-*, *-ness*)

**Types of bound morphemes:**
- **Prefix:** attaches before the root (*un-happy*, *re-write*)
- **Suffix:** attaches after the root (*happi-ness*, *walk-ing*)
- **Infix:** inserted within the root (rare in English, common in Tagalog)

### 3.2 Morphological Processes

**Inflection:** Changes the *grammatical form* of a word without creating a new lexeme. The word stays in the same syntactic category.
- Nouns: singular → plural (*cat → cats*)
- Verbs: tense/aspect (*walk → walked, walking, walks*)
- Adjectives: degree (*big → bigger → biggest*)

> **Key rule (exam favourite):** *"Inflectional morphemes never change the grammatical category (POS) of a word."*

**Derivation:** Creates a *new lexeme*, often changing the syntactic category.
- *child* (noun) → *childhood* (noun, but new meaning)
- *yellow* (adjective) → *yellowish* (adjective, diminutive meaning)
- *central* (adjective) → *centralise* (verb)
- *do* (verb) → *undo* (verb, opposite meaning)

**Compounding:** New words created by combining independent free morphemes.
- Noun + Noun: *girlfriend*, *blackbird*, *bookshelf*
- Verb + Noun: *guesswork*, *rainfall*
- Adjective + Noun: *blackboard*

**Example with root 'rain':**
- Inflection: *rains*, *rained*, *raining* (verb stays a verb)
- Derivation: *rainy* (adjective from noun), *rainfall* (noun from verb+noun)
- Compounding: *raincoat*, *rainbow*, *rainforest*

**Example with root 'drive':**
- Inflection: *drives*, *drove*, *driven*, *driving*
- Derivation: *driver* (noun from verb), *driveable* (adjective from verb)
- Compounding: *drivethrough*, *drive-in*

### 3.3 Stemming

**Definition:** Stemming is a crude, heuristic process that removes affixes from a word to produce a stem (not necessarily a real word).

**Purpose:** Reduce inflectional and some derivational variants to a common base form to improve recall in IR systems.

**The Porter Stemmer** (most widely used) applies a sequence of rules:

**Step 1a — Plurals:**
```
sses → ss    (caresses → caress)
ies  → i     (ponies → poni)
ss   → ss    (caress → caress)
s    → ∅     (cats → cat)
```

**Step 1b — Past tense, -ing:**
```
(m>0) eed → ee    (agreed → agree)
(*v*) ed  → ∅     (plastered → plaster)
(*v*) ing → ∅     (motoring → motor)
```

**Steps 2–5:** Further suffix stripping (e.g., *-ational → -ate*, *-ness → ∅*)

**Examples:**
```
fishing → fish
fished  → fish
fisher  → fisher  (not fish — the r blocks stripping)
argue   → argu
arguing → argu
```

**Python NLTK stemming:**
```python
from nltk.stem import PorterStemmer, LancasterStemmer

ps = PorterStemmer()
words = ["fishing", "fished", "arguing", "argued", "happily", "happiness"]
for w in words:
    print(f"{w:15} → {ps.stem(w)}")
# fishing         → fish
# fished          → fish
# arguing         → argu
# argued          → argu
# happily         → happili
# happiness       → happi
```

### 3.4 Lemmatisation

**Definition:** Lemmatisation produces the *canonical dictionary form* (lemma) of a word, using vocabulary and morphological analysis. The result is always a real word.

**Process:** Requires:
1. The word form
2. Its POS tag (to determine which paradigm to use)
3. A lexical database (e.g., WordNet)

**Examples:**
```
were   → be      (verb past tense → infinitive)
better → good    (adjective superlative → base)
ran    → run
dogs   → dog
```

### 3.5 Stemming vs. Lemmatisation Comparison Table

| Feature | Stemming | Lemmatisation |
|---------|----------|---------------|
| Output | Stem (may not be a real word) | Lemma (always a real word) |
| Method | Rule-based suffix stripping | Vocab lookup + morphological analysis |
| Speed | Fast | Slower |
| Accuracy | Lower (over/under-stem) | Higher |
| Requires POS? | No | Yes (for best results) |
| Requires lexicon? | No | Yes (e.g., WordNet) |
| *fishing* | fish | fish |
| *better* | better | good |
| *are* | are | be |
| *happily* | happili (Porter) | happily |
| **Best for** | IR, search engines | Tasks requiring real words (QA, MT) |

**Python spaCy lemmatisation:**
```python
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("The dogs were running and the children were happier")
for token in doc:
    print(f"{token.text:15} POS: {token.pos_:6}  Lemma: {token.lemma_}")
# dogs            POS: NOUN    Lemma: dog
# were            POS: AUX     Lemma: be
# running         POS: VERB    Lemma: run
# children        POS: NOUN    Lemma: child
# happier         POS: ADJ     Lemma: happy
```

### 3.6 When to Use Which

- **Use stemming** when: speed is critical; recall matters more than precision; you're building an IR index; exact word form is not needed
- **Use lemmatisation** when: the output needs to be a real word; POS information is available; for tasks like machine translation, question answering, or knowledge extraction

---

## 4. Corpora and Data

### 4.1 What is a Corpus?

A **corpus** (pl. *corpora*) is a large, principled collection of text (and sometimes speech) used for linguistic analysis or to train NLP models.

**Key properties of a good corpus:**
- Representativeness: covers the domain/language of interest
- Size: enough data for statistical reliability
- Annotation: labels that make it useful for supervised learning
- Provenance: known source, genre, date

### 4.2 Types of Corpora

| Type | Description | Example |
|------|-------------|---------|
| **Raw/unannotated** | Plain text, no labels | Web crawl data |
| **POS-annotated** | Words tagged with POS | Penn Treebank |
| **Treebank** | Parse trees for every sentence | Penn Treebank, Universal Dependencies |
| **Sense-tagged** | Words labelled with WordNet senses | SemCor |
| **NER-annotated** | Named entities labelled | CoNLL 2003 |
| **Sentiment-annotated** | Reviews with sentiment labels | SST, IMDB |
| **Parallel corpus** | Aligned texts in two+ languages | Europarl, UN corpus |
| **Domain-specific** | Medical, legal, scientific | PubMed, MIMIC |

### 4.3 NLTK Built-in Corpora

```python
import nltk
nltk.download('all')  # download all corpora

# Brown Corpus: 1 million words, 15 genres, POS-tagged
from nltk.corpus import brown
print(brown.categories())
# ['adventure', 'belles_lettres', 'editorial', ..., 'romance']
words_news = brown.words(categories='news')
tagged_news = brown.tagged_words(categories='news')

# Gutenberg: classic literature (plain text)
from nltk.corpus import gutenberg
print(gutenberg.fileids())  # ['austen-emma.txt', 'bible-kjv.txt', ...]
emma_words = gutenberg.words('austen-emma.txt')

# Reuters: news wire, 10,788 docs, 90 categories
from nltk.corpus import reuters

# WordNet: lexical database
from nltk.corpus import wordnet as wn
syns = wn.synsets('bank')
for s in syns[:3]:
    print(s.name(), s.definition())

# Frequency distribution
from nltk import FreqDist
fd = FreqDist(brown.words())
print(fd.most_common(10))
fd.plot(30)  # plot top 30 words
```

### 4.4 Annotation and Inter-Annotator Agreement

**Inter-annotator agreement (IAA):** The degree to which two (or more) independent annotators agree on a label.

**Cohen's Kappa (κ):**

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

Where:
- $P_o$ = observed agreement (proportion of items annotators agree on)
- $P_e$ = expected agreement by chance

| κ value | Interpretation |
|---------|----------------|
| < 0.20 | Slight agreement |
| 0.21–0.40 | Fair agreement |
| 0.41–0.60 | Moderate agreement |
| 0.61–0.80 | Substantial agreement |
| 0.81–1.00 | Almost perfect agreement |

**Worked example:**
Two annotators classify 100 sentences as Positive/Negative:
- Agree on Positive: 40; Agree on Negative: 30; Disagree: 30
- $P_o = (40+30)/100 = 0.70$
- Annotator A: 50 Pos, 50 Neg; Annotator B: 55 Pos, 45 Neg
- $P_e = (0.50 \times 0.55) + (0.50 \times 0.45) = 0.275 + 0.225 = 0.50$
- $\kappa = (0.70 - 0.50)/(1 - 0.50) = 0.20/0.50 = 0.40$ (Fair agreement)

### 4.5 Training, Validation, and Test Splits

| Split | Typical Size | Purpose |
|-------|-------------|---------|
| Training | 60–80% | Fit model parameters |
| Validation/Dev | 10–20% | Tune hyperparameters |
| Test | 10–20% | Final evaluation |

**Golden rule:** The test set must never be used during model development. Using test data for tuning leads to **data leakage** and over-optimistic performance estimates.

---

## 5. Information Retrieval

### 5.1 IR Fundamentals

**Information Retrieval (IR)** is the task of finding documents in a collection that satisfy an information need expressed as a query.

Use cases: web search, enterprise search, library catalogues, site search.

### 5.2 Boolean Retrieval

**Boolean model:** Documents either match a query or they don't — no ranking.

**Inverted index:** Maps each term to the list of document IDs containing it.

```
"python" → [doc1, doc3, doc7, doc12]
"NLP"    → [doc1, doc4, doc7, doc9]
"python AND NLP" → intersect → [doc1, doc7]
```

**Advantages:** Transparent, predictable, precise control
**Disadvantages:** No ranking, requires exact query formulation, poor recall for inexact matches

### 5.3 Vector Space Model

Each document and query is represented as a vector in a high-dimensional space where each dimension corresponds to a vocabulary term.

**Document ranking** by cosine similarity between the query vector and each document vector.

### 5.4 TF-IDF ⭐ (Calculation appears in every exam)

**Term Frequency (TF):** How often does term $t$ appear in document $d$?

$$\text{tf}_{t,d} = \text{count}(t, d)$$

Or with log normalisation (used in most IR systems):

$$\text{tf}_{t,d} = \begin{cases} 1 + \log_{10}(\text{count}(t,d)) & \text{if count}(t,d) > 0 \\ 0 & \text{otherwise} \end{cases}$$

**Inverse Document Frequency (IDF):** How rare is term $t$ across the corpus?

$$\text{idf}_t = \log_{10}\left(\frac{N}{\text{df}_t}\right)$$

Where:
- $N$ = total number of documents in the collection
- $\text{df}_t$ = number of documents containing term $t$

**TF-IDF score:**

$$w_{t,d} = \text{tf}_{t,d} \times \text{idf}_t = (1 + \log_{10}(\text{count}(t,d))) \times \log_{10}\left(\frac{N}{\text{df}_t}\right)$$

**Document score for query $q$:**

$$\text{Score}(q, d) = \sum_{t \in q} w_{t,d}$$

**Intuition:** TF rewards terms that appear frequently in a document; IDF penalises terms that appear in many documents (i.e., common, less informative terms like *the*).

### 5.5 TF-IDF Worked Example ⭐

**Corpus:** N = 3 documents

| | Doc1 | Doc2 | Doc3 |
|--|------|------|------|
| Content | "the cat sat on the mat" | "the dog sat" | "the cat and the dog" |

**Step 1: Count term frequencies**

| Term | Doc1 count | Doc2 count | Doc3 count | df |
|------|-----------|-----------|-----------|-----|
| cat  | 1 | 0 | 1 | 2 |
| sat  | 1 | 1 | 0 | 2 |
| dog  | 0 | 1 | 1 | 2 |
| mat  | 1 | 0 | 0 | 1 |
| the  | 2 | 1 | 2 | 3 |

**Step 2: Compute IDF** (N=3)

| Term | df | idf = log₁₀(3/df) |
|------|----|--------------------|
| cat  | 2  | log₁₀(3/2) = log₁₀(1.5) ≈ 0.176 |
| sat  | 2  | log₁₀(1.5) ≈ 0.176 |
| dog  | 2  | log₁₀(1.5) ≈ 0.176 |
| mat  | 1  | log₁₀(3/1) = log₁₀(3) ≈ 0.477 |
| the  | 3  | log₁₀(3/3) = log₁₀(1) = 0.000 |

**Note:** *the* has IDF = 0 because it appears in all documents — it carries no discriminative information.

**Step 3: Compute TF (log-normalised)**

| Term | Doc1 tf | Doc2 tf | Doc3 tf |
|------|---------|---------|---------|
| cat  | 1+log₁₀(1)=1 | 0 | 1+log₁₀(1)=1 |
| sat  | 1 | 1 | 0 |
| dog  | 0 | 1 | 1 |
| mat  | 1 | 0 | 0 |
| the  | 1+log₁₀(2)≈1.301 | 1 | 1+log₁₀(2)≈1.301 |

**Step 4: Compute TF-IDF**

| Term | Doc1 TF-IDF | Doc2 TF-IDF | Doc3 TF-IDF |
|------|------------|------------|------------|
| cat  | 1×0.176=**0.176** | 0×0.176=0 | 1×0.176=**0.176** |
| sat  | 1×0.176=**0.176** | 1×0.176=**0.176** | 0×0.176=0 |
| dog  | 0 | 1×0.176=**0.176** | 1×0.176=**0.176** |
| mat  | 1×0.477=**0.477** | 0 | 0 |
| the  | 1.301×0=0 | 1×0=0 | 1.301×0=0 |

**Observation:** *mat* has the highest TF-IDF in Doc1 — it appears only in Doc1, making it highly distinctive. *the* has TF-IDF = 0 everywhere.

**Query scoring:** Query = "cat mat"
- Score(q, Doc1) = TF-IDF(cat, Doc1) + TF-IDF(mat, Doc1) = 0.176 + 0.477 = **0.653**
- Score(q, Doc2) = 0 + 0 = **0**
- Score(q, Doc3) = 0.176 + 0 = **0.176**

Ranking: Doc1 > Doc3 > Doc2

**Exam trick for TF-IDF from the Sept 2024 paper:** *"A document containing 300 words with a term appearing 25 times; the term appears 400 times in a corpus of 50,000 words."*

In this question the corpus statistics are given in terms of total words, not document counts. The question is ambiguous — interpret as document-level statistics.
- TF = 25/300 ≈ 0.083 (raw TF, not log-normalised — use raw if not told otherwise)
- For IDF you'd need number of documents, not total words. If the question intends df = (total term occurrences in corpus) as a proxy, note the ambiguity and state your assumption clearly.

**The standard approach (always use this formula in CM3060):**
$$w_{t,d} = (1 + \log_{10} \text{tf}_{t,d}) \times \log_{10}(N / \text{df}_t)$$

### 5.6 Cosine Similarity ⭐

**Definition:** Measures the angle between two document vectors, independent of their magnitude.

$$\text{cos}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{|\vec{q}||\vec{d}|} = \frac{\sum_i q_i d_i}{\sqrt{\sum_i q_i^2} \cdot \sqrt{\sum_i d_i^2}}$$

**Range:** -1 to 1 (for non-negative TF-IDF vectors: 0 to 1). Higher = more similar.

**Worked Example:**

Query vector (TF-IDF): q = [cat: 0.5, dog: 0.3, mat: 0.2]
Doc1 vector (TF-IDF): d1 = [cat: 0.4, dog: 0.0, mat: 0.6]
Doc2 vector (TF-IDF): d2 = [cat: 0.6, dog: 0.4, mat: 0.0]

**Step 1:** Dot products
- q · d1 = (0.5×0.4) + (0.3×0.0) + (0.2×0.6) = 0.20 + 0.00 + 0.12 = 0.32
- q · d2 = (0.5×0.6) + (0.3×0.4) + (0.2×0.0) = 0.30 + 0.12 + 0.00 = 0.42

**Step 2:** Magnitudes
- |q|  = √(0.5²+0.3²+0.2²) = √(0.25+0.09+0.04) = √0.38 ≈ 0.616
- |d1| = √(0.4²+0.0²+0.6²) = √(0.16+0.00+0.36) = √0.52 ≈ 0.721
- |d2| = √(0.6²+0.4²+0.0²) = √(0.36+0.16+0.00) = √0.52 ≈ 0.721

**Step 3:** Cosine similarity
- cos(q, d1) = 0.32 / (0.616 × 0.721) = 0.32 / 0.444 ≈ **0.721**
- cos(q, d2) = 0.42 / (0.616 × 0.721) = 0.42 / 0.444 ≈ **0.946**

**Ranking:** Doc2 > Doc1 for this query.

### 5.7 BM25 (Best Match 25)

BM25 is a more sophisticated ranking function that accounts for document length and term saturation:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{idf}(t) \cdot \frac{\text{tf}(t,d) \cdot (k_1 + 1)}{\text{tf}(t,d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

Where:
- $k_1 \approx 1.2$–2.0: controls TF saturation
- $b \approx 0.75$: controls length normalisation
- $|d|$: document length; $\text{avgdl}$: average document length

BM25 outperforms plain TF-IDF in most IR benchmarks. Know the intuition: BM25 prevents a document from getting infinite score just because a term appears very frequently.

### 5.8 IR Evaluation Metrics ⭐

**Setting:** A system retrieves a set of documents for a query. Some are relevant, some are not.

| | Retrieved | Not Retrieved |
|--|-----------|--------------|
| **Relevant** | True Positive (TP) | False Negative (FN) |
| **Not Relevant** | False Positive (FP) | True Negative (TN) |

$$\text{Precision} = \frac{TP}{TP + FP}$$

*Of all documents I retrieved, what fraction are relevant?*

$$\text{Recall} = \frac{TP}{TP + FN}$$

*Of all relevant documents that exist, what fraction did I retrieve?*

$$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Precision–Recall trade-off:** Retrieving more documents generally increases recall but decreases precision (you start retrieving irrelevant documents). A good system achieves high precision at high recall levels.

**Mean Average Precision (MAP):** Averages the precision at each recall point across multiple queries.

**NDCG (Normalised Discounted Cumulative Gain):** Used when documents have graded relevance (not just binary). Rewards highly relevant documents ranked higher.

**Worked Confusion Matrix Example (from 2024 exam):**

| | Predicted Positive | Predicted Negative | Total |
|--|-------------------|-------------------|-------|
| **Actual Positive** | 10 | 20 | 30 |
| **Actual Negative** | 30 | 40 | 70 |
| **Total** | 40 | 60 | 100 |

- TP = 10, FN = 20, FP = 30, TN = 40
- Accuracy = (TP+TN)/(total) = (10+40)/100 = **50%**
- Precision = TP/(TP+FP) = 10/40 = **25%**
- Recall = TP/(TP+FN) = 10/30 = **33.3%**
- F1 = 2×(0.25×0.333)/(0.25+0.333) = 2×0.0833/0.583 = 0.1666/0.583 ≈ **28.6%**

**From 2022-Sep exam (confusion matrix):**

| | Predicted+ | Predicted- |
|--|-----------|-----------|
| True+ | 10 | 20 |
| True- | 30 | 40 |

- TP=10, FP=30, FN=20, TN=40
- Accuracy = (10+40)/100 = **50%**
- Precision = 10/40 = **0.25**
- Recall = 10/30 = **0.33**
- F1 = 2×(0.25×0.33)/(0.25+0.33) = 0.165/0.58 ≈ **0.284**

**Why accuracy fails on imbalanced data:** If 90% of emails are spam, a classifier that always predicts "spam" achieves 90% accuracy but 0% recall for ham. F1 would be: Precision = 0.9, Recall = 1.0, F1 = 0.947 — but the system is useless for ham detection.

---

## 6. Language Models and N-grams

### 6.1 What is a Language Model?

A **language model** assigns probabilities to sequences of words. It answers: *"How likely is this sequence of words in this language?"*

$$P(W) = P(w_1, w_2, \ldots, w_n)$$

**Applications:** spell checking, speech recognition, MT, text generation, next-word prediction.

### 6.2 The Chain Rule

By the chain rule of probability:

$$P(w_1, w_2, \ldots, w_n) = \prod_{i=1}^{n} P(w_i | w_1, w_2, \ldots, w_{i-1})$$

This is exact but computationally intractable for long sequences.

### 6.3 The Markov Assumption

**N-gram language models** approximate using the Markov assumption: the probability of a word depends only on the preceding N−1 words.

- **Unigram:** $P(w_i)$ — no context
- **Bigram:** $P(w_i | w_{i-1})$
- **Trigram:** $P(w_i | w_{i-2}, w_{i-1})$

$$P(w_1, \ldots, w_n) \approx \prod_{i=1}^{n} P(w_i | w_{i-N+1}, \ldots, w_{i-1})$$

### 6.4 Maximum Likelihood Estimation (MLE)

**Bigram MLE:**

$$P(w_i | w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i)}{\text{count}(w_{i-1})}$$

**Unigram MLE:**

$$P(w_i) = \frac{\text{count}(w_i)}{\text{total word count in corpus}}$$

### 6.5 Worked Bigram Calculation ⭐

**From 2022-Sep past paper — Corpus counts (100M word corpus):**

| n-gram | count |
|--------|-------|
| my | 883,614 |
| beliefs | 80,891 |
| chimneys | 21 |
| my beliefs | 378 |
| my chimneys | 0 |

**Vocabulary size V = 1,234,567**

**(a) P(beliefs) using MLE (no smoothing):**

$$P(\text{beliefs}) = \frac{80,891}{100,000,000} = \frac{80891}{100000000}$$

**(a) P(beliefs | my) using MLE:**

$$P(\text{beliefs} | \text{my}) = \frac{\text{count(my beliefs)}}{\text{count(my)}} = \frac{378}{883,614}$$

**(b) P(chimneys | my) using Laplace smoothing:**

$$P_L(\text{chimneys} | \text{my}) = \frac{\text{count(my chimneys)} + 1}{\text{count(my)} + V} = \frac{0 + 1}{883,614 + 1,234,567} = \frac{1}{2,118,181}$$

**(c) P(chimneys | my) using add-k smoothing (k=0.1):**

$$P_{0.1}(\text{chimneys} | \text{my}) = \frac{\text{count(my chimneys)} + 0.1}{\text{count(my)} + 0.1 \times V} = \frac{0 + 0.1}{883,614 + 0.1 \times 1,234,567} = \frac{0.1}{883,614 + 123,456.7} = \frac{0.1}{1,007,070.7}$$

**From 2023-March paper (very similar structure — know this pattern):**

| n-gram | count |
|--------|-------|
| my | 876,543 |
| thoughts | 87,654 |
| carpets | 21 |
| my thoughts | 378 |
| my carpets | 0 |

- P(thoughts) = 87,654 / 100,000,000
- P(thoughts | my) = 378 / 876,543
- P(carpets | my) with Laplace = 1 / (876,543 + 1,234,567) = 1 / 2,111,110
- P(carpets | my) with k=0.2 = 0.2 / (876,543 + 0.2 × 1,234,567) = 0.2 / (876,543 + 246,913.4) = 0.2 / 1,123,456.4

### 6.6 Sentence Probability Calculation

**Setup:** Vocabulary = {I, am, Sam, like, green, eggs}, N = 3 sentences in training data:
```
<s> I am Sam </s>
<s> Sam I am </s>
<s> I am green eggs </s>
```

Add start `<s>` and end `</s>` tokens. Count bigrams:

| Bigram | Count |
|--------|-------|
| `<s>` I | 2 |
| `<s>` Sam | 1 |
| I am | 3 |
| am Sam | 1 |
| Sam `</s>` | 1 |
| Sam I | 1 |
| am green | 1 |
| green eggs | 1 |
| eggs `</s>` | 1 |

Count unigrams: `<s>`=3, I=3, am=3, Sam=2, green=1, eggs=1, `</s>`=3

**P("I am Sam") using bigram model:**

$$P(\text{I am Sam}) = P(\text{I}|\langle s\rangle) \times P(\text{am}|\text{I}) \times P(\text{Sam}|\text{am}) \times P(\langle/s\rangle|\text{Sam})$$

$$= \frac{2}{3} \times \frac{3}{3} \times \frac{1}{3} \times \frac{1}{2} = \frac{2}{3} \times 1 \times \frac{1}{3} \times \frac{1}{2} = \frac{2}{18} = \frac{1}{9} \approx 0.111$$

**With Laplace smoothing** (V = 6 unique words, + 2 boundary symbols = 8):

$$P_L(\text{am}|\text{I}) = \frac{3+1}{3+8} = \frac{4}{11}$$

### 6.7 Data Sparsity Problem

With a vocabulary of size V, a bigram model has V² possible bigrams. Most will never appear in training data, giving P = 0. A zero probability for any n-gram makes the entire sentence probability zero.

### 6.8 Smoothing Methods

| Method | Formula | Pros | Cons |
|--------|---------|------|------|
| **Laplace (add-1)** | Count+1 / (total + V) | Simple | Over-smooths (steals too much probability from seen events) |
| **Add-k** | Count+k / (total + kV), 0<k<1 | Flexible | k must be tuned |
| **Good-Turing** | Use count(n+1) items to estimate count(n) | Principled | Complex to implement |
| **Backoff (Katz)** | If trigram unseen, use bigram; if bigram unseen, use unigram | Handles sparsity well | Needs discount factor |
| **Interpolation** | λ₁P(w) + λ₂P(w|w₋₁) + λ₃P(w|w₋₂,w₋₁) | Combines all orders | λs must sum to 1 and be tuned |
| **Kneser-Ney** | Uses continuation probability | State of the art | Complex formula |

**Kneser-Ney key idea:** Instead of raw frequency, use *continuation probability* — how many different contexts does a word appear in? A word like *Francisco* always follows *San*, so its unigram continuation count is very low. *Kong* always follows *Hong*. This gives better backoff estimates.

**Comparing smoothing methods (for exam Q "compare and contrast"):**
- Laplace is crude but pedagogically clear; always increases probability of unseen events
- Add-k is more flexible (k < 1 steals less probability mass)
- Good-Turing is based on the frequency-of-frequency statistics
- Kneser-Ney is empirically the best-performing but most complex
- Backoff vs interpolation: backoff falls back sequentially; interpolation always mixes all orders

### 6.9 Perplexity ⭐

**Definition:** Perplexity (PP) is the inverse probability of the test set, normalised by the number of words. It measures how well the language model predicts the test data.

$$\text{PP}(W) = P(w_1, w_2, \ldots, w_N)^{-1/N}$$

For a bigram model:

$$\text{PP}(W) = \left(\prod_{i=1}^{N} \frac{1}{P(w_i|w_{i-1})}\right)^{1/N}$$

**Lower perplexity = better model** (more certain predictions).

**Intuition:** If perplexity = 100, the model is as confused as if choosing uniformly among 100 options at each step.

**Perplexity of a random digit string:** There are 10 equally likely digits (0–9). PP = 10. (This is why the exam answer is 10.)

**Exam question from mock:** "Perplexity scores of 234, 123, and 876 for unigram, bigram, trigram respectively — does this sound right?"

**Answer:** No, this does not sound right. A trigram model should have *lower* perplexity than a bigram, which should be lower than a unigram, because the trigram conditions on more context and makes better predictions. The correct ordering should be: PP(unigram) > PP(bigram) > PP(trigram). A trigram perplexity of 876 when the bigram achieves 123 indicates either a very poorly estimated trigram model (possibly due to extreme data sparsity with inadequate smoothing) or an error.

**Python NLTK language model:**
```python
from nltk.lm import MLE, Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.util import bigrams

# Training data
train_sents = [["I", "am", "Sam"], ["Sam", "I", "am"], ["I", "am", "green"]]

# Build bigram MLE model
n = 2
train_data, padded_sents = padded_everygram_pipeline(n, train_sents)
model = MLE(n)
model.fit(train_data, padded_sents)

print(model.score("am", ["I"]))     # P(am | I)
print(model.perplexity(["I", "am", "Sam"]))
```

---

## 7. Part-of-Speech Tagging

### 7.1 What is POS Tagging?

**Part-of-speech (POS) tagging** assigns a grammatical category (noun, verb, adjective, etc.) to each word in a sentence. It is a prerequisite for many downstream NLP tasks (parsing, NER, lemmatisation).

**Example:**
```
The/DT quick/JJ brown/JJ fox/NN jumps/VBZ over/IN the/DT lazy/JJ dog/NN ./.
```

### 7.2 Penn Treebank POS Tagset

The most widely used English tagset in NLP. Know all these tags for the exam:

| Tag | Description | Example |
|-----|-------------|---------|
| CC | Coordinating conjunction | and, but, or |
| CD | Cardinal number | one, 2024 |
| DT | Determiner | the, a, an |
| EX | Existential *there* | there is |
| FW | Foreign word | déjà vu |
| IN | Preposition or subordinating conj | in, of, that |
| JJ | Adjective | big, green |
| JJR | Adjective, comparative | bigger |
| JJS | Adjective, superlative | biggest |
| LS | List item marker | 1., 2., a. |
| MD | Modal | can, will, should |
| NN | Noun, singular or mass | cat, water |
| NNS | Noun, plural | cats |
| NNP | Proper noun, singular | London |
| NNPS | Proper noun, plural | Vikings |
| PDT | Predeterminer | all, both |
| POS | Possessive ending | 's |
| PRP | Personal pronoun | I, he, she |
| PRP$ | Possessive pronoun | my, his |
| RB | Adverb | quickly |
| RBR | Adverb, comparative | more quickly |
| RBS | Adverb, superlative | most quickly |
| RP | Particle | up (in *pick up*) |
| SYM | Symbol | %, & |
| TO | *to* | to |
| UH | Interjection | uh, wow |
| VB | Verb, base form | run, eat |
| VBD | Verb, past tense | ran, ate |
| VBG | Verb, gerund/present participle | running, eating |
| VBN | Verb, past participle | run, eaten |
| VBP | Verb, non-3rd person singular present | run (I run) |
| VBZ | Verb, 3rd person singular present | runs, eats |
| WDT | Wh-determiner | which, that |
| WP | Wh-pronoun | who, what |
| WP$ | Possessive wh-pronoun | whose |
| WRB | Wh-adverb | where, when |
| PU | Punctuation | . , ! ? |

**Singular noun vs. mass noun vs. plural noun (exam question 2024):**
- Singular noun (countable): *dog, book, idea* — can be preceded by *a/an*
- Mass noun (uncountable): *water, sand, information* — cannot normally be pluralised
- Plural noun: *dogs, books, ideas*

All three use the tag **NN** (singular/mass) or **NNS** (plural).

**Exam POS tagging exercise (from 2024 paper):**

*"They wind back the clock, while we chase after the wind."*

```
They/PRP  wind/VBP  back/RP  the/DT  clock/NN  ,/PU
while/IN  we/PRP  chase/VBP  after/IN  the/DT  wind/NN  ./.
```

**Note:** *wind* is tagged VBP (verb) in the first use and NN (noun) in the second. This demonstrates **lexical ambiguity** — the same word form has different POS depending on context.

**Exam POS tagging exercise (from 2022-Sep):**

*"Rick and Francis wrote a song with three chords."*
```
Rick/NNP  and/CC  Francis/NNP  wrote/VBD  a/DT  song/NN  with/IN  three/CD  chords/NNS  ./.
```

*"It was played in a very large hall near their studio."*
```
It/PRP  was/VBD  played/VBN  in/IN  a/DT  very/RB  large/JJ  hall/NN  near/IN  their/PRP$  studio/NN  ./.
```

### 7.3 Rule-Based POS Tagging

**Brill tagger (transformation-based):**
1. Start by assigning each word its most common tag
2. Apply transformation rules that change a tag based on context
3. Rules are learned from annotated data by greedy search

**Example rules:**
- If a word is tagged VBD and the next word is tagged TO, change tag to VB
- If a word is tagged NN and the previous word is MD, change tag to VB

### 7.4 HMM-Based POS Tagging

**Hidden Markov Model (HMM):** A sequence model where we observe words (the observations) and want to infer the POS tags (the hidden states).

**Two probability tables:**

**Transition probabilities:** P(tag_i | tag_{i-1}) — how likely is tag j to follow tag i?

$$A_{ij} = P(t_j | t_i)$$

**Emission probabilities:** P(word | tag) — how likely is this word given the tag?

$$B_{j}(w) = P(w | t_j)$$

**Goal:** Find the sequence of tags that maximises P(tags | words).

**Computing emission probabilities from training data:**

$$P(w | t) = \frac{\text{count}(w, t)}{\text{count}(t)}$$

**Computing transition probabilities:**

$$P(t_i | t_{i-1}) = \frac{\text{count}(t_{i-1}, t_i)}{\text{count}(t_{i-1})}$$

### 7.5 Computing Emission Probabilities (Exam Exercise)

**From 2023-March paper — Training data:**
```
plane/NN  flying/VBG  is/VBZ  stressful/JJ
flying/VBG  planes/NNS  are/VBP  dangerous/JJ
I/PRP  saw/VBD  Lauren/NNP  flying/VBG  planes/NNS
She/PRP  planes/NNS  doors/NNS
```

**Step 1: Count (word, tag) pairs**

| Word | Tag | Count |
|------|-----|-------|
| plane | NN | 1 |
| flying | VBG | 3 |
| is | VBZ | 1 |
| stressful | JJ | 1 |
| planes | NNS | 3 |
| are | VBP | 1 |
| dangerous | JJ | 1 |
| I | PRP | 1 |
| saw | VBD | 1 |
| Lauren | NNP | 1 |
| She | PRP | 1 |
| doors | NNS | 1 |

**Step 2: Count tag frequencies**

| Tag | Total count |
|-----|-------------|
| NN | 1 |
| VBG | 3 |
| VBZ | 1 |
| JJ | 2 |
| NNS | 4 |
| VBP | 1 |
| PRP | 2 |
| VBD | 1 |
| NNP | 1 |

**Step 3: Compute P(word | tag)**

- P(flying | VBG) = 3/3 = **1.0**
- P(planes | NNS) = 3/4 = **0.75**
- P(doors | NNS) = 1/4 = **0.25**
- P(stressful | JJ) = 1/2 = **0.5**
- P(dangerous | JJ) = 1/2 = **0.5**
- P(I | PRP) = 1/2 = **0.5**
- P(She | PRP) = 1/2 = **0.5**

### 7.6 The Viterbi Algorithm ⭐

**Purpose:** Efficiently find the most probable sequence of hidden states (POS tags) given a sequence of observations (words). Uses dynamic programming.

**Pseudocode:**

```
Input: words w₁...wₙ, tags T, transition A, emission B, initial π
Output: best tag sequence

# Initialisation
for each tag t in T:
    viterbi[t][1] = π[t] × B[t](w₁)
    backpointer[t][1] = 0

# Recursion
for position i from 2 to n:
    for each tag t in T:
        viterbi[t][i] = max over s in T: (viterbi[s][i-1] × A[s][t] × B[t](wᵢ))
        backpointer[t][i] = argmax over s in T: (viterbi[s][i-1] × A[s][t])

# Termination
best_final_tag = argmax over t in T: viterbi[t][n]

# Backtrack
best_sequence[n] = best_final_tag
for i from n-1 down to 1:
    best_sequence[i] = backpointer[best_sequence[i+1]][i+1]

return best_sequence
```

**Worked Viterbi Trace — 3 words, 3 tags:**

**Setup:**
- Words: "Janet will back"
- Tags: NNP (proper noun), MD (modal), VB (verb)
- Transition probabilities A:

| From↓ To→ | NNP | MD | VB |
|-----------|-----|----|----|
| `<S>` | 0.4 | 0.3 | 0.3 |
| NNP | 0.1 | 0.5 | 0.4 |
| MD | 0.1 | 0.05 | 0.85 |
| VB | 0.5 | 0.1 | 0.4 |

- Emission probabilities B:

| Tag↓ Word→ | Janet | will | back |
|------------|-------|------|------|
| NNP | 0.8 | 0.01 | 0.01 |
| MD | 0.01 | 0.7 | 0.1 |
| VB | 0.01 | 0.1 | 0.5 |

**Viterbi Trellis:**

**Position 1 ("Janet"):**
- v(NNP,1) = 0.4 × 0.8 = **0.320**
- v(MD, 1) = 0.3 × 0.01 = **0.003**
- v(VB, 1) = 0.3 × 0.01 = **0.003**

**Position 2 ("will"):**
- v(NNP,2) = max(v(NNP,1)×A[NNP→NNP], v(MD,1)×A[MD→NNP], v(VB,1)×A[VB→NNP]) × B(NNP,"will")
  = max(0.320×0.1, 0.003×0.1, 0.003×0.5) × 0.01
  = max(0.032, 0.0003, 0.0015) × 0.01 = 0.032 × 0.01 = **0.00032** (from NNP)
- v(MD, 2) = max(0.320×0.5, 0.003×0.05, 0.003×0.1) × 0.7
  = max(0.160, 0.00015, 0.0003) × 0.7 = 0.160 × 0.7 = **0.112** (from NNP)
- v(VB, 2) = max(0.320×0.4, 0.003×0.85, 0.003×0.4) × 0.1
  = max(0.128, 0.00255, 0.0012) × 0.1 = 0.128 × 0.1 = **0.0128** (from NNP)

**Position 3 ("back"):**
- v(NNP,3) = max(0.00032×0.1, 0.112×0.1, 0.0128×0.5) × 0.01
  = max(0.000032, 0.0112, 0.0064) × 0.01 = 0.0112 × 0.01 = **0.000112** (from MD)
- v(MD, 3) = max(0.00032×0.5, 0.112×0.05, 0.0128×0.1) × 0.1
  = max(0.00016, 0.0056, 0.00128) × 0.1 = 0.0056 × 0.1 = **0.00056** (from MD)
- v(VB, 3) = max(0.00032×0.4, 0.112×0.85, 0.0128×0.4) × 0.5
  = max(0.000128, 0.0952, 0.00512) × 0.5 = 0.0952 × 0.5 = **0.0476** (from MD)

**Best final tag:** VB (0.0476 is highest at position 3)
**Backtrack:** VB ← MD (position 2 best predecessor for VB at 3 was MD) ← NNP (position 1 best predecessor for MD at 2 was NNP)

**Result:** NNP MD VB → "Janet/NNP will/MD back/VB" ✓

**ASCII Trellis:**
```
          Janet         will          back
NNP:  [0.320,NNP] → [0.00032,NNP] → [0.000112,MD]
MD:   [0.003,NNP] → [0.112,NNP]  → [0.00056,MD]
VB:   [0.003,NNP] → [0.0128,NNP] → [0.0476,MD] ←← BEST
                                          ↑
                                    Backtrack: VB←MD←NNP
```

---

## 8. Formal Grammars and Parsing

### 8.1 Context-Free Grammars (CFGs)

**Definition:** A CFG consists of:
- A set of **terminal symbols** (words): e.g., *{cat, sat, the, on, mat}*
- A set of **non-terminal symbols** (categories): e.g., *{S, NP, VP, PP, Det, NN, VBD, IN}*
- A **start symbol**: S
- A set of **production rules**: non-terminal → sequence of terminals/non-terminals

**Common rules:**
```
S  → NP VP
NP → Det NN
NP → Det NN PP
NP → NNP
PP → IN NP
VP → VBD NP
VP → VBD NP PP
VP → VBD
Det → the | a
NN  → cat | mat | dog
VBD → sat | saw
IN  → on | with
NNP → London | Mary
```

### 8.2 Parse Trees

A **parse tree** (or phrase structure tree) shows the hierarchical structure of a sentence according to a grammar.

**Example: "The cat sat on the mat"**
```
         S
        / \
       NP   VP
      / \   |  \
    Det  NN VBD  PP
    |    |  |   /  \
   the  cat sat IN   NP
               |   /   \
               on Det    NN
                  |      |
                 the    mat
```

**Ambiguity in parse trees:** *"I saw the man on the hill with a telescope"* has multiple parses:

**Interpretation 1:** I used a telescope to see the man
```
         S
        / \
       NP   VP
       |   /   \
       I  VBD   NP
          |    /   \
         saw  NP    PP
             / \   /  \
           Det  NN IN  NP
           |    |  |  /   \
          the  man with  Det  NN
                        /     \
                       a   telescope
```

**Interpretation 2:** The man on the hill had a telescope
(PP "with a telescope" attaches to "the man on the hill")

This is **prepositional phrase attachment ambiguity** — a classic problem in parsing.

### 8.3 Shortcomings of CFGs

1. **Structural ambiguity:** Multiple parse trees for one sentence — the grammar assigns no preference
2. **Lack of lexical sensitivity:** PP attachment depends on the specific verb and noun, not just structure (*"eat spaghetti with a fork"* vs *"eat spaghetti with meatballs"*)
3. **Word order variations** are hard to handle cleanly with rules alone
4. **Long-distance dependencies:** e.g., *"The horse raced past the barn fell"*
5. **Scale:** Hand-writing rules for all of English is impractical

**Solution — Probabilistic CFGs (PCFGs):**

### 8.4 Probabilistic CFG (PCFG)

Each production rule is assigned a probability. All rules expanding the same non-terminal must sum to 1.

```
S  → NP VP         [1.0]
NP → Det NN        [0.5]
NP → NNP           [0.3]
NP → Det NN PP     [0.2]
VP → VBD NP        [0.6]
VP → VBD NP PP     [0.4]
```

**Parse probability:** product of probabilities of all rules used in the derivation.

**Selected parse = the parse with the highest probability.**

Probabilities estimated from a **treebank** (annotated corpus of parse trees).

### 8.5 Chomsky Normal Form (CNF)

**Definition:** A CFG is in CNF if every rule is of the form:
- A → B C (a non-terminal expands to exactly two non-terminals), **or**
- A → w (a non-terminal expands to exactly one terminal)

**Why:** The CYK algorithm requires CNF.

**Converting to CNF:**
1. Eliminate ε-productions (rules of form A → ε)
2. Eliminate unit productions (A → B)
3. Binarise long rules: A → B C D E → A → B X, X → C Y, Y → D E
4. Move terminals in mixed rules: A → B c → A → B C', C' → c

**Example:**
```
Original:  VP → VBD NP PP
CNF step:  VP → VBD X
           X  → NP PP
```

### 8.6 CYK Parsing Algorithm ⭐

**Purpose:** Given a grammar in CNF and a sentence, find all possible parse trees (or determine if the sentence is grammatical).

**Complexity:** O(n³ × |G|) where n = sentence length, |G| = grammar size.

**CYK Table notation:** table[i][j] = set of non-terminals that span words i to j.

**Worked CYK Example:**

**Grammar (CNF):**
```
S  → NP VP
NP → Det N
VP → V NP
Det → the
N   → dog | man
V   → bit
```

**Sentence:** "the dog bit the man" (indices 1–5)

| | the(1) | dog(2) | bit(3) | the(4) | man(5) |
|--|--------|--------|--------|--------|--------|
| **[1,1]** | Det | | | | |
| **[2,2]** | | N | | | |
| **[3,3]** | | | V | | |
| **[4,4]** | | | | Det | |
| **[5,5]** | | | | | N |
| **[1,2]** | NP | | | | |
| **[2,3]** | | — | | | |
| **[3,4]** | | | — | | |
| **[4,5]** | | | | NP | |
| **[1,3]** | — | | | | |
| **[2,4]** | | — | | | |
| **[3,5]** | | | VP | | |
| **[1,4]** | — | | | | |
| **[2,5]** | | — | | | |
| **[1,5]** | **S** | | | | |

**S is in [1,5] — sentence is grammatical!** The parse derives as: S → NP[1,2] VP[3,5] → Det[1,1] N[2,2] V[3,3] NP[4,5] → "the dog bit the man"

### 8.7 Dependency Grammar

**Dependency grammar** represents syntactic structure as binary, asymmetric relationships between a **head** word and its **dependents**, rather than as phrase structure.

**Key properties:**
- No phrase nodes — just word-to-word relationships
- Each word (except root) has exactly one head
- Relationships are labelled (subject, object, modifier, etc.)
- The head of a sentence is typically the main verb

**Example:** "I saw the man on the hill with the telescope"
```
saw
├── I (nsubj)
├── man (dobj)
│   ├── the (det)
│   └── hill (prep)
│       └── on (case)
│           └── the (det)
└── telescope (prep)
    └── with (case)
        └── the (det)
```

**Projective graph:** No crossing arcs (all arcs can be drawn above the sentence without crossing). English is mostly projective; some languages (e.g., German, Czech) have many non-projective constructions.

**Coordination in dependency parsing** is particularly difficult. In *"I bought books and magazines"*, *and* coordinates *books* and *magazines* — both are objects of *bought*. The head of a coordination is typically the first conjunct or the conjunction itself, but different annotation schemes disagree.

**Ellipsis** is another challenge: *"John likes coffee and Mary [likes] tea"* — the verb *likes* is omitted in the second conjunct but its dependency structure still exists. Parsing ellipsis requires recovering the elided material.

### 8.8 Recursion in CFGs

**Direct recursion:** A rule where the left-hand non-terminal appears on the right-hand side.
```
NP → NP PP    (NP directly contains NP)
```
This generates: "the cat on the mat near the window..."

**Indirect recursion:** A non-terminal appears on the right-hand side through a chain of rules.
```
S  → NP VP
VP → V S     (VP can expand to V then another S, which expands to NP VP, ...)
```
This generates embedded sentences: *"I think that she said that he knew that..."*

**Example grammar with both types of recursion:**
```
S   → NP VP         (base rule)
NP  → Det NN        (base NP)
NP  → NP PP         (direct recursion: NP contains NP)
VP  → VBD NP        (base VP)
VP  → VBD S         (indirect recursion: VP contains S, S contains VP)
PP  → IN NP
Det → the | a
NN  → dog | bone | park
VBD → chased | said
IN  → in | near
```

**Grammatical sentences:**
1. "the dog chased a bone" — no recursion
2. "the dog in the park chased a bone" — direct NP recursion (NP → NP PP)
3. "the dog said the cat chased a bone" — indirect recursion (VP → VBD S)

**Ungrammatical sentence (rejected):**
- *"dog chased"* — no determiner, rejects *NN VP* since NP requires Det NN

---

## 9. Named Entity Recognition

### 9.1 NER Definition and Entity Types

**Named Entity Recognition (NER)** identifies and classifies named entities in text — proper nouns that refer to real-world entities.

**Standard entity types:**
| Type | Description | Examples |
|------|-------------|---------|
| PER/PERSON | People | *Barack Obama*, *Einstein* |
| ORG | Organisations | *Google*, *University of London* |
| GPE | Geo-political entity | *France*, *New York* |
| LOC | Geographic location (non-GPE) | *Mount Everest*, *Pacific Ocean* |
| DATE | Dates and periods | *March 2024*, *the 1990s* |
| TIME | Times | *3:00 PM* |
| MONEY | Monetary values | *$50 million* |
| PERCENT | Percentages | *25%* |
| PRODUCT | Products/artefacts | *iPhone 15* |
| EVENT | Events | *World Cup 2022* |

### 9.2 IOB/BIO Tagging Scheme ⭐

NER is treated as a **sequence labelling** problem. Each token is assigned one of:
- **B-TYPE:** Beginning of a named entity of TYPE
- **I-TYPE:** Inside (continuing) a named entity of TYPE
- **O:** Outside any named entity

**Example:**
```
Sentence: "Jordan visited Jordan this summer"

Jordan/B-PER  visited/O  Jordan/B-GPE  this/O  summer/O
```

Note: The word *Jordan* is tagged B-PER (person) the first time and B-GPE (geopolitical entity) the second time. This illustrates the importance of context — NER must disambiguate based on surrounding words ("visited" is a clue that Jordan is a traveller; the second Jordan follows "visited" as a destination).

**NER BIO example from 2022-Sep:**
```
Noshin has a car with a sunroof and a set of alloy wheels.

[B-PER Noshin] [O has] [O a] [B-NP car] [O with] [O a] [B-NP sunroof] 
[O and] [O a] [B-NP set] [O of] [B-NP alloy] [I-NP wheels] [O .]
```

Wait — the exam used BIO for noun groups, not named entities. Let me show both:

**BIO for noun groups (chunking):**
```
Noshin/B-NP  has/O  a/B-NP  car/I-NP  with/O  a/B-NP  sunroof/I-NP  
and/O  a/B-NP  set/I-NP  of/O  alloy/B-NP  wheels/I-NP  .
```

**BIO for NER:**
```
United/B-ORG  Airlines/I-ORG  Holding/I-ORG  is/O  an/O  ORG/O
```

### 9.3 NER Approaches

**Rule-based NER:**
- Uses gazetteers (lists of known entity names), regular expressions, capitalisation patterns
- High precision on known entities; poor recall for novel entities
- Example: match `[A-Z][a-z]+` followed by `[A-Z][a-z]+` as a person name

**Statistical NER (HMM, CRF):**
- Conditional Random Fields (CRFs) are the most popular pre-neural approach
- Features: word itself, capitalisation, word shape (*Xxxx*), POS tag, surrounding words, prefixes/suffixes
- CRF learns the conditional probability of a label sequence given the input

**Evaluation — entity-level F1:**
- A predicted entity is correct only if both its span boundaries AND type are correct
- This is stricter than token-level accuracy

### 9.4 NER with spaCy

```python
import spacy
nlp = spacy.load("en_core_web_sm")

text = "Jordan visited Jordan this summer to see the Dead Sea."
doc = nlp(text)

for ent in doc.ents:
    print(f"{ent.text:20} | {ent.label_:10} | {spacy.explain(ent.label_)}")

# Jordan               | PERSON     | People, including fictional
# Jordan               | GPE        | Countries, cities, states
# this summer          | DATE       | Absolute or relative dates or periods
# the Dead Sea         | LOC        | Non-GPE locations, mountain ranges, bodies of water

# Visualise
from spacy import displacy
displacy.render(doc, style="ent")
```

**Coreference resolution** (related task): Identifying when different expressions refer to the same entity.
- *"John told Peter that **his** friends were coming"* — does *his* refer to John or Peter?
- Not directly handled by spaCy's basic model; requires a coreference resolution component

---

## 10. Word Representations

### 10.1 One-Hot Encoding

Each word is represented as a vector of length |V| (vocabulary size) with a 1 in its position and 0s everywhere else.

**Vocabulary:** {cat, dog, mat, sat, the} → |V| = 5

```
cat = [1, 0, 0, 0, 0]
dog = [0, 1, 0, 0, 0]
mat = [0, 0, 1, 0, 0]
```

**Shortcomings (exam MCQ):**
- Vectors are **very sparse** (mostly zeros)
- Vectors are **very high-dimensional** (|V| can be 50,000–500,000)
- No notion of similarity — all words are equidistant from each other
- *cat* · *dog* = 0 even though they are semantically similar

### 10.2 Bag of Words (BoW)

A document is represented as a vector of term frequencies. No word order information.

**Example sentence:** "Stars are distant suns in the night sky"

**Vocabulary (no stopwords):** {stars, distant, suns, night, sky}

```
BoW vector = [stars:1, distant:1, suns:1, night:1, sky:1]
```

**Limitations:**
- Ignores word order: *"This film was exciting and not boring"* vs *"This film was boring and not exciting"* — same BoW representation
- Ignores grammar and syntax
- High-dimensional, sparse

### 10.3 Distributed Word Representations (Embeddings)

Words are represented as dense, low-dimensional vectors where similar words have similar vectors.

**The Distributional Hypothesis:** *"You shall know a word by the company it keeps"* (Firth, 1957). Words with similar meanings tend to appear in similar contexts.

**Advantages over one-hot:**
- Captures semantic similarity
- Dense (50–300 dimensions, not |V| dimensions)
- Most values are non-zero
- Enables generalisation — model sees *cat*, can make predictions about *dog*
- Better for downstream classification (fewer weights to learn)

### 10.4 Word2Vec ⭐ (High-yield)

Word2Vec (Mikolov et al., 2013) trains word embeddings using a neural network with two architectures:

**CBOW (Continuous Bag of Words):**
- **Input:** context words (window around the target)
- **Output:** predict the target word in the middle
- *"The ___ sat on the mat"* → predict *cat*

**Skip-gram:**
- **Input:** the target word
- **Output:** predict each context word in the window
- *"cat"* → predict {*the, sat, on, the*}

**Training objective:** Maximise the log probability of context words given the target (or vice versa).

**Which is better?**
- CBOW: faster, better for frequent words, good for smaller corpora
- Skip-gram: slower but better for rare words, works well on large corpora

**Training tricks:**
- **Negative sampling:** Rather than computing softmax over all |V| words, treat training as binary classification (is this a real context pair or a noise pair?)
- **Hierarchical softmax:** Use a binary tree over vocabulary to speed up computation

**Note from mock exam:** The question asked which are "training algorithms" — the answer is CBOW and Skip-gram. Negative sampling and hierarchical softmax are *optimisation techniques*, not separate architectures.

### 10.5 Word2Vec: Analogical Reasoning

The famous property: vector arithmetic captures semantic relationships.

$$\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$$

**ASCII Diagram of vector arithmetic:**

```
Vector Space (simplified 2D projection):

          king ●
                \
                 \  difference = "royalty"
                  \
          man ●   queen ●
               \        
                woman ●

king - man + woman ≈ queen
```

Other examples:
- Paris - France + Italy ≈ Rome (capital relationships)
- walked - walk + run ≈ ran (verb tense)
- bigger - big + cold ≈ colder (adjective comparison)

### 10.6 GloVe (Global Vectors)

GloVe (Pennington et al., 2014) trains embeddings using global co-occurrence statistics from the entire corpus (vs. Word2Vec's local context windows). GloVe minimises the difference between the dot product of two word vectors and the log of their co-occurrence count.

**Semantic drift:** Over time, word meanings change (e.g., *"gay"* shifted from "happy" to referring to sexual orientation; *"cloud"* acquired a computing meaning). If a Word2Vec model is retrained on data from different periods, the same word will have different vector representations. This is called semantic drift or semantic shift.

### 10.7 Comparison Table

| Method | Dimensionality | Density | Captures Similarity | Order | Computation |
|--------|---------------|---------|---------------------|-------|-------------|
| One-hot | |V| (~50k+) | Very sparse | No | No | Trivial |
| BoW | |V| | Sparse | No | No | Trivial |
| TF-IDF | |V| | Sparse | No | No | Light |
| Word2Vec | 50–300 | Dense | Yes | No | Moderate |
| GloVe | 50–300 | Dense | Yes | No | Moderate |
| BERT | 768–1024 | Dense | Yes (contextual) | Yes | Heavy |

**Key difference:** Word2Vec/GloVe produce **static** embeddings (one vector per word regardless of context). BERT produces **contextual** embeddings (*"bank"* gets different vectors in different sentences).

### 10.8 Python Word2Vec with gensim

```python
from gensim.models import Word2Vec
from nltk.corpus import brown
import nltk

nltk.download('brown')
sentences = brown.sents()  # list of tokenised sentences

# Train Word2Vec
model = Word2Vec(
    sentences=sentences,
    vector_size=100,    # embedding dimension
    window=5,           # context window size
    min_count=5,        # ignore words with freq < 5
    workers=4,
    sg=1                # 1=skip-gram, 0=CBOW
)

# Similarity
print(model.wv.most_similar('king', topn=5))
# [('queen', 0.78), ('prince', 0.76), ...]

# Analogy
result = model.wv.most_similar(
    positive=['king', 'woman'], 
    negative=['man'], 
    topn=1
)
print(result)  # [('queen', 0.75)]

# Save and load
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")
```

---

## 11. Text Classification

### 11.1 The Classification Pipeline

```
Raw Text → Preprocessing → Feature Extraction → Classifier → Label
                               (BoW / TF-IDF
                                / embeddings)
```

### 11.2 Naïve Bayes Classifier ⭐

**Bayes' Theorem:**

$$P(c|d) = \frac{P(d|c) \cdot P(c)}{P(d)}$$

For classification, we want:

$$\hat{c} = \arg\max_{c \in C} P(c|d) = \arg\max_{c \in C} P(d|c) \cdot P(c)$$

**Bag-of-words representation** + **conditional independence assumption** (naïve):

$$P(d|c) = \prod_{i=1}^{n} P(w_i|c)$$

**Full formula:**

$$\hat{c}_{NB} = \arg\max_{c} \underbrace{P(c)}_{\text{prior}} \cdot \prod_{i \in \text{positions}} \underbrace{P(w_i|c)}_{\text{likelihood}}$$

**MLE estimates:**

$$P(c) = \frac{N_c}{N} \quad \text{(proportion of documents in class c)}$$

$$P(w|c) = \frac{\text{count}(w, c)}{\sum_{w'} \text{count}(w', c)}$$

**With Laplace smoothing:**

$$P(w|c) = \frac{\text{count}(w, c) + 1}{\sum_{w'} \text{count}(w', c) + |V|}$$

**In log space** (to avoid underflow from multiplying many small probabilities):

$$\hat{c} = \arg\max_{c} \log P(c) + \sum_i \log P(w_i|c)$$

### 11.3 Naïve Bayes Worked Example ⭐

**Task:** Classify a movie review as Positive or Negative.

**Training data:**

| Review | Class |
|--------|-------|
| "great movie love it" | Positive |
| "amazing film awesome" | Positive |
| "terrible movie hate it" | Negative |
| "boring film bad" | Negative |

**Counts:**

| Word | Positive count | Negative count |
|------|---------------|----------------|
| great | 1 | 0 |
| movie | 1 | 1 |
| love | 1 | 0 |
| it | 1 | 1 |
| amazing | 1 | 0 |
| film | 1 | 1 |
| awesome | 1 | 0 |
| terrible | 0 | 1 |
| hate | 0 | 1 |
| boring | 0 | 1 |
| bad | 0 | 1 |

- Total positive tokens: 7 | Total negative tokens: 7
- P(Positive) = 2/4 = 0.5 | P(Negative) = 2/4 = 0.5
- |V| = 11

**Laplace-smoothed probabilities:**

P(great | Pos) = (1+1)/(7+11) = 2/18 = 1/9
P(movie | Pos) = (1+1)/(7+11) = 2/18 = 1/9
P(great | Neg) = (0+1)/(7+11) = 1/18
P(movie | Neg) = (1+1)/(7+11) = 2/18 = 1/9

**Classify: "great movie"**

Score(Pos) = log(0.5) + log(1/9) + log(1/9) = log(0.5) - log(9) - log(9)
Score(Neg) = log(0.5) + log(1/18) + log(1/9) = log(0.5) - log(18) - log(9)

Score(Pos) > Score(Neg) → classified as **Positive**

*Great* contributes more evidence for positive because P(great|Pos) = 1/9 > P(great|Neg) = 1/18.

### 11.4 Feature Engineering for Text Classification

**Bag of Words (BoW):** Binary or count features for each vocabulary term.

**TF-IDF features:** Downweights common words, upweights rare distinctive words.

**N-gram features:** Include bigrams and trigrams to capture local word order. E.g., *"not good"* as a bigram captures negation that BoW misses.

**Character n-grams:** Useful for handling typos, morphologically rich languages.

**Converting BoW to feature vector (exam question from mock 2021):**

*"Suppose you are doing bag-of-words text classification on a document. The raw input is a single string. Describe the process to convert to a feature vector."*

1. **Tokenise** the string into word tokens
2. **Normalise:** lowercase, remove punctuation
3. **Remove stop words** (optionally)
4. **Stem or lemmatise** (optionally)
5. **Build vocabulary** from training documents
6. **Count occurrences** of each vocabulary term in the document
7. **Create vector** of length |V| where each position holds the count (or TF-IDF weight) of that term
8. **Normalise vector** (optional: L2 normalisation for cosine similarity)

### 11.5 Python scikit-learn Text Classification

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Training data
train_docs = [
    "great movie love it",
    "amazing film awesome",
    "terrible movie hate it",
    "boring film bad"
]
train_labels = ["pos", "pos", "neg", "neg"]

# Pipeline: TF-IDF vectoriser + Naive Bayes
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

pipeline.fit(train_docs, train_labels)

# Predict
test_docs = ["great film", "horrible movie"]
predictions = pipeline.predict(test_docs)
print(predictions)  # ['pos', 'neg']
```

### 11.6 Overfitting in Text Classifiers

**Overfitting:** Model performs well on training data but poorly on unseen test data. The model has memorised training instances rather than learning generalisable patterns.

**Signs:** Training accuracy >> test accuracy.

**Strategies to address overfitting:**
1. **Regularisation:** L1 (sparsity) or L2 (weight decay) penalties on model parameters
2. **More training data:** The most effective remedy
3. **Feature selection:** Remove noisy or rare features (e.g., minimum document frequency threshold)
4. **Dimensionality reduction:** PCA, SVD (LSA) to compress feature space
5. **Simpler model:** Fewer parameters, lower-order n-grams
6. **Cross-validation:** Use k-fold CV to detect overfitting during development
7. **Dropout** (for neural models)
8. **Early stopping** (for neural models)

**Pointwise Mutual Information (PMI):**

$$\text{PMI}(x, y) = \log_2 \frac{P(x, y)}{P(x) P(y)}$$

Measures how much more (or less) two events co-occur than expected by chance. Positive PMI → words co-occur more than expected (good collocations). Applied to sentiment: if *"excellent"* co-occurs strongly with positive reviews, PMI(excellent, positive) >> 0.

---

## 12. Sentiment Analysis

### 12.1 What is Sentiment Analysis?

**Sentiment analysis** (or opinion mining) automatically identifies the polarity (positive/negative/neutral) or sentiment of text.

**Levels of analysis:**
- Document level: Is this review positive or negative overall?
- Sentence level: Is this sentence positive or negative?
- Aspect level: Is the camera quality of this phone positive or negative?

### 12.2 Lexicon-Based Approaches

**Approach:** Use a sentiment lexicon (dictionary mapping words to polarity scores). Sum scores to get document sentiment.

**Example lexicons:**
- VADER (Valence Aware Dictionary and sEntiment Reasoner) — designed for social media
- SentiWordNet — extends WordNet with positive/negative/objective scores
- LIWC (Linguistic Inquiry and Word Count)

**Advantages:** No training data needed; interpretable; fast
**Disadvantages:** Domain-dependent (a lexicon for restaurant reviews may not work for financial news); doesn't handle negation, sarcasm, intensifiers well

### 12.3 ML-Based Approaches

Train a classifier (Naïve Bayes, Logistic Regression, SVM, neural) on labelled sentiment data.

**Handling negation:** One baseline technique — add the suffix NOT_ to every word between a negation and the following punctuation:

*"I didn't like this movie"* → *"I didn't NOT_like NOT_this NOT_movie"*

This way *NOT_like* appears in negative reviews and carries negative sentiment signal.

**Binary multinomial Naïve Bayes for sentiment:** Rather than using term frequency, use binary features (word appears or not). This prevents documents with repeated terms from dominating. Count is important in IR but less so in sentiment.

### 12.4 Challenges in Sentiment Analysis

1. **Negation:** *"This is not good"* — surface features predict positive, but meaning is negative
2. **Irony/sarcasm:** *"Oh great, another Monday"* — surface features predict positive
3. **Domain shift:** A model trained on movie reviews fails on medical texts
4. **Comparative sentiment:** *"This is better than I expected"* — better than what?
5. **Implicit sentiment:** *"The battery died after 2 hours"* — no sentiment word, but clearly negative
6. **Multi-polarity:** *"The food was great but the service was terrible"*
7. **Aspect identification:** Which aspect is being evaluated?

### 12.5 VADER with Python NLTK

```python
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

sentences = [
    "This movie is amazing!",
    "I did not like this film at all.",
    "The acting was okay but the plot was terrible.",
    "Great product for £15. Good sound, very easy connection!",  # from 2024 exam
]

for s in sentences:
    scores = sia.polarity_scores(s)
    print(f"{s[:40]:40} | compound: {scores['compound']:+.3f}")
    
# VADER returns: neg, neu, pos (proportions), compound (−1 to +1)
# compound > 0.05 → positive; < −0.05 → negative; else neutral
```

### 12.6 Extracting Sentiment from Product Reviews (Exam Q 2024)

*"What are the main tasks in extracting sentiment from data such as product reviews?"*

**Main challenges and tasks:**
1. **Preprocessing:** Handle slang, abbreviations (*"comfy"*, *"10/10"*), emojis (😂), ALL CAPS (*"HUGE"*), typographical errors
2. **Aspect extraction:** Identify what feature is being evaluated (*"sound"*, *"battery life"*, *"comfort"*)
3. **Opinion extraction:** Identify opinion words (*"amazing"*, *"HUGE"*)
4. **Negation handling:** *"no wonder there are no face on views"*
5. **Target identification:** Who/what is being evaluated?
6. **Aggregation:** Combining multiple aspects into an overall score
7. **Domain adaptation:** Reviews for electronics differ from reviews for restaurants

---

## 13. Information Extraction

### 13.1 IE Pipeline

```
Raw Text
    ↓
Named Entity Recognition (NER)
    ↓
Relation Extraction
    ↓
Event Extraction
    ↓
Coreference Resolution
    ↓
Structured Knowledge Base
```

### 13.2 Relation Extraction

**Goal:** Identify semantic relationships between named entities.

**Example:** *"Steve Jobs co-founded Apple in 1976"*
→ Relation: FOUNDED(Steve Jobs, Apple)
→ Relation: FOUNDED_IN(Apple, 1976)

**Approaches:**
1. **Pattern-based (hand-written):** E.g., *[ORG] was founded by [PER]* → FOUNDED_BY
2. **Supervised:** Train a classifier on labelled (entity1, relation, entity2) triples
3. **Semi-supervised / bootstrapping:** Start with seed instances, find patterns, use patterns to find more instances
4. **Distant supervision:** Use a knowledge base (Freebase, Wikidata) to automatically label training data
5. **Open information extraction (OpenIE):** Extract all relations without predefined schema

### 13.3 Coreference Resolution

**Definition:** Determining which expressions in a text refer to the same real-world entity.

**Example:** *"John told Peter that his friends were coming over"*
- *his* could refer to *John* or *Peter*
- Resolving coreference requires world knowledge and discourse context

**Mention types:**
- **Named:** *Barack Obama*
- **Nominal:** *the president*
- **Pronominal:** *he*, *they*, *it*

**Approaches:**
1. **Rule-based:** Hobbs' algorithm (syntactic proximity)
2. **Mention-pair models:** Classify each pair of mentions as coreferent or not
3. **Entity-mention models:** Track entities across the document
4. **End-to-end neural:** Jointly learn mention detection and clustering (Lee et al., 2017)

---

## 14. Statistical NLP Evaluation

### 14.1 Precision, Recall, F1 (Detailed) ⭐

These metrics appear in essentially every exam paper. Master them.

**Binary classification:**

$$\text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Recall} = \frac{TP}{TP + FN}$$
$$\text{F1} = \frac{2 \cdot P \cdot R}{P + R} = \frac{2TP}{2TP + FP + FN}$$
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**F_β measure:** Weights recall β times as much as precision:

$$F_\beta = \frac{(1+\beta^2) \cdot P \cdot R}{\beta^2 \cdot P + R}$$

F1 is $F_\beta$ with β=1 (equal weight). F2 weights recall higher (useful when missing a relevant item is costly, e.g., medical diagnosis).

### 14.2 Multi-Class Evaluation ⭐

**From 2023-March and 2023-Sep exams — 3-class confusion matrix:**

The 2023-Sep exam (tweet sentiment: Positive/Neutral/Negative):

|  | Pred Pos | Pred Neu | Pred Neg |
|--|---------|---------|---------|
| **True Pos** | 40 | 6 | 4 |
| **True Neu** | 8 | 36 | 4 |
| **True Neg** | 6 | 2 | 14 |

**Per-class precision and recall:**

For **Positive:**
- TP = 40
- FP = 8 + 6 = 14 (Neutral and Negative predicted as Positive)
- FN = 6 + 4 = 10 (Positive predicted as Neutral or Negative)
- Precision = 40/(40+14) = 40/54 ≈ **0.741**
- Recall = 40/(40+10) = 40/50 = **0.800**
- F1 = 2×(0.741×0.800)/(0.741+0.800) = 1.186/1.541 ≈ **0.769**

For **Neutral:**
- TP = 36
- FP = 6 + 2 = 8
- FN = 8 + 4 = 12
- Precision = 36/(36+8) = 36/44 ≈ **0.818**
- Recall = 36/(36+12) = 36/48 = **0.750**
- F1 = 2×(0.818×0.750)/(0.818+0.750) = 1.227/1.568 ≈ **0.783**

For **Negative:**
- TP = 14
- FP = 4 + 4 = 8
- FN = 6 + 2 = 8
- Precision = 14/(14+8) = 14/22 ≈ **0.636**
- Recall = 14/(14+8) = 14/22 ≈ **0.636**
- F1 = 2×(0.636×0.636)/(0.636+0.636) = 0.809/1.272 ≈ **0.636**

**Micro-average precision:**

Pool all TP, FP, FN across classes:
- Total TP = 40 + 36 + 14 = 90
- Total FP = 14 + 8 + 8 = 30 (wait — let's recount)
  - FP_Pos = 8+6 = 14; FP_Neu = 6+2 = 8; FP_Neg = 4+4 = 8; Total FP = 30
- Micro-P = Total TP / (Total TP + Total FP) = 90 / (90+30) = 90/120 = **0.75**

Note: For micro-average in multi-class, micro-precision = micro-recall = accuracy when all classes are considered.

Total predictions = 40+8+6+6+36+2+4+4+14 = 120 (sum of all entries)
Correct = 40+36+14 = 90
Micro-average precision = accuracy = 90/120 = **0.75**

**Macro-average precision:**

Simple average across classes:
Macro-P = (0.741 + 0.818 + 0.636) / 3 = 2.195/3 ≈ **0.732**

**Why they differ:** Macro-average treats all classes equally regardless of size. Micro-average weights each instance equally, so large classes dominate. Here the Negative class is smallest (22 items) and has lowest precision (0.636); macro-average is pulled down by this, while micro-average is dominated by Positive (largest class, 0.741 precision).

### 14.3 BLEU Score for Machine Translation ⭐

**BLEU (Bilingual Evaluation Understudy)** measures the quality of machine-translated text by comparing it to one or more reference translations.

**Key idea:** Count n-gram overlaps between the hypothesis (MT output) and reference translation(s).

**Modified precision for n-grams:**

For each n-gram in the hypothesis, count the maximum number of times it appears in any reference:

$$p_n = \frac{\sum_{\text{ngram} \in \hat{y}} \min(\text{count}(\text{ngram}, \hat{y}), \text{count}(\text{ngram}, y))}{\sum_{\text{ngram} \in \hat{y}} \text{count}(\text{ngram}, \hat{y})}$$

**BLEU-4 (standard):**

$$\text{BLEU} = \text{BP} \times \exp\left(\sum_{n=1}^{4} w_n \log p_n\right)$$

Where:
- $w_n = 1/4$ (uniform weights over 1-, 2-, 3-, 4-grams)
- **BP (Brevity Penalty):** penalises short translations

$$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ e^{1-r/c} & \text{if } c \leq r \end{cases}$$

Where c = hypothesis length, r = reference length.

**BLEU Worked Example:**

- Reference: *"the cat sat on the mat"* (length = 6)
- Hypothesis: *"the cat is on the mat"* (length = 6)

**Unigram precision (p₁):**
Hypothesis unigrams: the(2), cat(1), is(1), on(1), mat(1) — total = 6
Match with reference: the→min(2,2)=2, cat→min(1,1)=1, is→min(1,0)=0, on→min(1,1)=1, mat→min(1,1)=1
p₁ = (2+1+0+1+1)/6 = 5/6 ≈ 0.833

**Bigram precision (p₂):**
Hypothesis bigrams: (the,cat)(1), (cat,is)(1), (is,on)(1), (on,the)(1), (the,mat)(1) — total = 5
Reference bigrams: (the,cat)(1), (cat,sat)(1), (sat,on)(1), (on,the)(1), (the,mat)(1)
Matches: (the,cat)=1, (on,the)=1, (the,mat)=1 — 3 matches
p₂ = 3/5 = 0.600

**BP:** c = r = 6, so BP = 1.0

**BLEU-2 (using only 1- and 2-grams with equal weights):**

$$\text{BLEU-2} = 1.0 \times \exp(0.5 \times \log(5/6) + 0.5 \times \log(3/5))$$
$$= \exp(0.5 \times (-0.182) + 0.5 \times (-0.511))$$
$$= \exp(-0.091 - 0.255) = \exp(-0.347) \approx 0.707$$

**Limitations of BLEU:**
- Multiple valid translations → use multiple references when available
- Doesn't capture meaning, only surface overlap
- Bad at evaluating fluency vs. adequacy separately
- Not well-correlated with human judgement at sentence level (better at corpus level)

### 14.4 Perplexity (Full Definition)

Already covered in Section 6.9. Key formula for exam:

$$\text{PP}(W) = P(w_1 w_2 \ldots w_N)^{-1/N}$$

For bigram model:
$$\text{PP}(W) = \left(\prod_{i=1}^N \frac{1}{P(w_i|w_{i-1})}\right)^{1/N} = 2^{H}$$

Where $H$ is the per-word entropy of the language model.

### 14.5 Intrinsic vs. Extrinsic Evaluation

**Intrinsic evaluation:** Measures performance on a specific intermediate NLP task in isolation, using a dedicated benchmark dataset.
- Examples: perplexity for language models; accuracy on a POS tagging test set; F1 on NER; BLEU for MT

**Extrinsic evaluation:** Measures performance on a downstream application that uses the NLP component.
- Examples: accuracy of a QA system that relies on a language model; click-through rate of a search engine; task completion rate of a dialogue system

**Trade-off:** Intrinsic evaluation is cheaper and faster; extrinsic evaluation tells you whether the component actually helps the real application. A component can improve intrinsically but not extrinsically (e.g., a better POS tagger may not improve a downstream IE system if that system doesn't use POS tags).

### 14.6 Accuracy vs. F1 (When to Use Which)

**Use accuracy when:**
- Classes are balanced (roughly equal numbers of each class)
- All types of error are equally costly

**Use F1 (or precision/recall separately) when:**
- Classes are imbalanced (e.g., spam detection: 99% not spam, 1% spam)
- False positives and false negatives have different costs
- The positive class is rare but important (medical diagnosis, fraud detection)

**Example (from 2024 exam):** A naive classifier that predicts "positive" for every review in a 70%/30% positive/negative dataset achieves 70% accuracy. This sounds good but has 0% recall for negative reviews (F1 for negative class = 0). This is why accuracy alone is misleading for imbalanced datasets.

---

## 15. NLP with Python (NLTK and spaCy)

### 15.1 Complete NLTK Pipeline

```python
import nltk
nltk.download(['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 
               'words', 'stopwords', 'wordnet'])

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.chunk import ne_chunk
from nltk import FreqDist, bigrams, trigrams

# --- FULL PIPELINE ---
raw_text = """
Natural language processing involves machines processing human language.
Dr. Smith can't believe how quickly NLP has advanced.
"""

# 1. Sentence tokenisation
sentences = sent_tokenize(raw_text.strip())
print("SENTENCES:", sentences)

# 2. Word tokenisation
all_tokens = []
for sent in sentences:
    tokens = word_tokenize(sent)
    all_tokens.extend(tokens)
print("TOKENS:", all_tokens[:10])

# 3. POS tagging
tagged = pos_tag(all_tokens)
print("TAGGED:", tagged[:10])

# 4. Normalisation (lowercase, remove punctuation)
from string import punctuation
normalised = [t.lower() for t in all_tokens if t not in punctuation]

# 5. Stop word removal
stop_words = set(stopwords.words('english'))
filtered = [t for t in normalised if t not in stop_words]
print("FILTERED:", filtered)

# 6. Stemming
ps = PorterStemmer()
stemmed = [ps.stem(t) for t in filtered]
print("STEMMED:", stemmed)

# 7. Lemmatisation (using POS for accuracy)
def get_wordnet_pos(tag):
    """Convert Penn Treebank tag to WordNet POS."""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default

wnl = WordNetLemmatizer()
lemmatised = [wnl.lemmatize(word, get_wordnet_pos(tag)) 
              for word, tag in tagged 
              if word.lower() not in stop_words and word not in punctuation]
print("LEMMATISED:", lemmatised)

# 8. Named Entity Recognition
ne_tree = ne_chunk(tagged)
for subtree in ne_tree:
    if hasattr(subtree, 'label'):
        entity = ' '.join(leaf[0] for leaf in subtree.leaves())
        print(f"NE: {entity} ({subtree.label()})")

# 9. Frequency distribution
fd = FreqDist(filtered)
print("TOP 10:", fd.most_common(10))

# 10. Bigrams
bg = list(bigrams(filtered))
bg_fd = FreqDist(bg)
print("TOP BIGRAMS:", bg_fd.most_common(5))
```

### 15.2 TF-IDF with scikit-learn

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

documents = [
    "the cat sat on the mat",
    "the dog sat",
    "the cat and the dog"
]

# Compute TF-IDF
vectorizer = TfidfVectorizer(
    use_idf=True,
    smooth_idf=False,      # don't add 1 to df (standard formula)
    sublinear_tf=True,     # use 1+log(tf) instead of tf
    norm='l2'              # normalise vectors
)

tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

# Display as matrix
import pandas as pd
df = pd.DataFrame(tfidf_matrix.toarray(), 
                  columns=feature_names,
                  index=['Doc1', 'Doc2', 'Doc3'])
print(df.round(3))

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(tfidf_matrix)
print("Cosine similarity matrix:")
print(sim_matrix.round(3))
```

### 15.3 Complete spaCy Pipeline

```python
import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

text = "Dr. Matthew Yee-King is Goldsmiths, University of London's Programme Director."

doc = nlp(text)

# Tokens and POS
print("\n--- TOKENS & POS ---")
for token in doc:
    print(f"{token.text:15} | POS: {token.pos_:8} | TAG: {token.tag_:6} | "
          f"Lemma: {token.lemma_:12} | Stop: {token.is_stop} | Dep: {token.dep_}")

# Named entities
print("\n--- NAMED ENTITIES ---")
for ent in doc.ents:
    print(f"{ent.text:30} | {ent.label_:10} | {spacy.explain(ent.label_)}")

# Dependency parse
print("\n--- DEPENDENCY PARSE ---")
for token in doc:
    print(f"{token.text:12} → {token.dep_:12} → head: {token.head.text}")

# Noun chunks (shallow parsing)
print("\n--- NOUN CHUNKS ---")
for chunk in doc.noun_chunks:
    print(f"{chunk.text:25} | root: {chunk.root.text} | root dep: {chunk.root.dep_}")

# Sentence segmentation
print("\n--- SENTENCES ---")
for sent in doc.sents:
    print(sent.text)
```

### 15.4 NLTK Corpus Access

```python
import nltk
from nltk.corpus import brown, reuters, gutenberg, wordnet as wn

# Brown corpus
print(brown.categories())
news_sents = brown.sents(categories='news')
news_tagged = brown.tagged_sents(categories='news')

# Conditional frequency distribution
from nltk import ConditionalFreqDist, ConditionalProbDist, MLEProbDist

# Distribution of modal verbs across genres
cfd = ConditionalFreqDist(
    (genre, word.lower())
    for genre in ['news', 'romance']
    for word in brown.words(categories=genre)
)
print(cfd['news']['the'], cfd['romance']['the'])

# WordNet
synsets = wn.synsets('bank')
for syn in synsets[:3]:
    print(syn.name(), ':', syn.definition())
    print('  Examples:', syn.examples())
    print('  Lemmas:', [l.name() for l in syn.lemmas()])
    
# Path similarity (0–1, 1 = identical)
cat = wn.synset('cat.n.01')
dog = wn.synset('dog.n.01')
print("cat-dog similarity:", cat.path_similarity(dog))
```

### 15.5 Bigram Language Model with NLTK

```python
from nltk.lm import MLE, Laplace, KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize, word_tokenize

# Prepare training data from Emma (Jane Austen)
text = gutenberg.raw('austen-emma.txt')
sentences = [word_tokenize(s.lower()) for s in sent_tokenize(text)]

n = 2  # bigram model
train_data, padded_sents = padded_everygram_pipeline(n, sentences)

# MLE bigram model
model = MLE(n)
model.fit(train_data, padded_sents)

# Score a bigram
print(model.score("was", ["it"]))      # P(was | it)
print(model.logscore("was", ["it"]))   # log probability

# Generate text
print(" ".join(model.generate(20, random_seed=3)))

# Perplexity on test sentence
from nltk.lm.preprocessing import padded_everygram_pipeline
test_sent = [word_tokenize("it was a beautiful day")]
test_data, _ = padded_everygram_pipeline(n, test_sent)
print("Perplexity:", model.perplexity(["it", "was", "a"]))
```

---

## 16. Regular Expressions

### 16.1 What are Regular Expressions Used For in NLP?

Regular expressions (regex) are formal pattern-matching tools based on the algebra of regular languages. In NLP, they are used for:
- **Tokenisation:** splitting text on whitespace, punctuation patterns
- **Pattern matching:** finding dates, phone numbers, URLs, email addresses
- **Text normalisation:** replacing patterns (e.g., remove punctuation)
- **Rule-based NER:** matching patterns for named entities
- **Spell checking:** edit distance combined with regex patterns

Limitation: regex cannot handle context-dependent patterns — they cannot remember what they saw earlier in the string (no counting, no recursion). For that, you need at least context-free grammars.

### 16.2 Key Regex Syntax

| Pattern | Meaning | Example |
|---------|---------|---------|
| `.` | Any character except newline | `a.b` matches "axb", "a7b" |
| `*` | Zero or more of preceding | `ab*c` matches "ac", "abc", "abbc" |
| `+` | One or more of preceding | `ab+c` matches "abc", "abbc" but not "ac" |
| `?` | Zero or one of preceding | `colou?r` matches "color" or "colour" |
| `{n}` | Exactly n repetitions | `\d{4}` matches "2024" |
| `{n,m}` | Between n and m repetitions | `\d{2,4}` matches "24", "2024" |
| `[abc]` | Character class | `[aeiou]` matches any vowel |
| `[^abc]` | Negated character class | `[^0-9]` matches non-digit |
| `[a-z]` | Character range | `[a-zA-Z]` matches any letter |
| `\d` | Digit (= `[0-9]`) | `\d+` matches integers |
| `\w` | Word character (= `[a-zA-Z0-9_]`) | `\w+` matches words |
| `\s` | Whitespace | `\s+` split on whitespace |
| `^` | Start of string (or line with MULTILINE) | `^Hello` |
| `$` | End of string | `world$` |
| `\|` | Disjunction (alternation) | `cat\|dog` matches "cat" or "dog" |
| `()` | Grouping (also captures) | `(ab)+` matches "ab", "abab" |
| `(?:...)` | Non-capturing group | `(?:ab)+` groups without capturing |
| `\1`, `\2` | Back-reference to captured group | `(\w+) \1` matches repeated words |

### 16.3 Disjunction, Grouping, Precedence

**Disjunction** (|): Matches either the pattern on the left or the right.
- `cat|dog` → matches "cat" or "dog"
- `(cat|dog) food` → matches "cat food" or "dog food"

**Grouping** (()): Groups sub-expressions; also captures the match.
- `(ab)+` → matches "ab", "abab", "ababab"

**Precedence** (operator priority, high to low):
1. Parentheses `()`
2. Quantifiers `* + ? {n,m}`
3. Sequence (concatenation): `ab` means a followed by b
4. Alternation `|`: lowest precedence

**Example:** `cat|dog+` means `cat` OR `dog+` (not `ca(t|d)og+`). To match "cat" or "dog" exactly: `(cat|dog)`.

### 16.4 Greedy Matching

By default, quantifiers (`*`, `+`, `?`, `{n,m}`) are **greedy** — they match as much text as possible.

**Example:** Pattern `<.*>` on `<b>text</b>` → matches `<b>text</b>` (the entire string, not just `<b>`).

**Problem:** This can cause over-matching in NLP tokenisation and extraction tasks.

**Solution — Non-greedy (lazy) quantifiers:** Add `?` after the quantifier: `*?`, `+?`, `??`

`<.*?>` on `<b>text</b>` → matches `<b>` only.

Alternatively, use negated character classes: `<[^>]*>` matches from `<` to the first `>`.

### 16.5 Exam Regex Exercises ⭐

**"Any string that contains at least four digits":**
```
\d.*\d.*\d.*\d
# or more elegantly:
.*\d.*\d.*\d.*\d.*
# or:
(?:.*\d){4}.*
```

**"Any string that ends with at least four digits":**
```
.*\d{4}$
```

**"Any string that starts with one uppercase character, ends with either two digits or three vowels":**
```
^[A-Z].*((\d{2})|([aeiouAEIOU]{3}))$
```

**"Any string that starts with one uppercase character, ends with 3 digits or 2 vowels" (from 2022-Sep):**
```
^[A-Z].*(\d{3}|[aeiouAEIOU]{2})$
```

**Are `^\d\d\d\d-\d\d-\d\d$` and `^\d{4}-\d{2}-\d{2}$` equivalent?** (from 2024 Sep exam)

**Yes**, they are functionally equivalent for matching YYYY-MM-DD date format. Both match exactly four digits, a hyphen, two digits, a hyphen, and two digits. The `{n}` quantifier is syntactic sugar for repeating `\d` exactly n times. The only difference is readability — `\d{4}` is more concise and arguably clearer.

**Validating timestamp `20:00, 2023/05/07`:**
```
\d{2}:\d{2},\s\d{4}/\d{2}/\d{2}
```

**Identifying URLs like `www.abc.com`:**
```
https?://[\w.-]+\.[a-zA-Z]{2,}(/\S*)?
# or for simple www-style:
www\.[\w.-]+\.[a-zA-Z]{2,}
```

**Identifying email like `info@abc.com`:**
```
[\w.+-]+@[\w-]+\.[\w.-]+
```

**Are `^[a-zA-Z][a-zA-Z][a-zA-Z]$` and `^[a-zA-Z]{3}$` equivalent?** (from 2023-Sep)
**Yes** — `{3}` is shorthand for exactly 3 repetitions of the preceding pattern.

### 16.6 Python Regex

```python
import re

text = "Dr. Matthew Yee-King is Goldsmiths, University of London's Programme Director."

# Find all words starting with uppercase
print(re.findall(r'\b[A-Z][a-z]+', text))

# Remove URLs
text_with_url = "Visit us at www.amazon.com for deals! Email: info@amazon.com"
clean = re.sub(r'https?://\S+|www\.\S+', '<URL>', text_with_url)
clean = re.sub(r'\S+@\S+\.\S+', '<EMAIL>', clean)
print(clean)

# Date validation
dates = ["2024-03-11", "24-3-11", "2024/03/11"]
pattern = r'^\d{4}-\d{2}-\d{2}$'
for d in dates:
    print(f"{d}: {bool(re.match(pattern, d))}")

# Tokenise: split on whitespace and punctuation
tokens = re.findall(r'\b\w+\b', text.lower())
print(tokens)

# Capture groups
phone_pattern = r'\((\d{3})\)\s*(\d{3})-(\d{4})'
match = re.search(phone_pattern, "Call (416) 555-1234 now")
if match:
    area, prefix, number = match.groups()
    print(f"Area: {area}, Prefix: {prefix}, Number: {number}")
```

---

## 17. Exam Strategy

### 17.1 How to Answer Algorithm Trace Questions

**For Viterbi:**
1. Draw the trellis clearly (words as columns, tags as rows)
2. Show the initialisation step at t=1
3. For each subsequent position, show the computation for each tag: which previous tag maximises the path, and multiply by emission probability
4. Show the backpointer at each cell
5. Identify the best final tag, then backtrack explicitly

**Template answer:**
```
At position i, tag T:
  Max(viterbi[s][i-1] × A[s→T]) for all s, then × B[T](word_i)
  = max(val_A × A[A→T], val_B × A[B→T], val_C × A[C→T]) × B[T](word_i)
  = max(x, y, z) × emission = VALUE (from PREV_TAG)
```

**For CYK:**
1. Write the sentence with word indices (1 to n)
2. Fill diagonal (single words → terminals → apply lexical rules)
3. Fill each longer span by trying all split points k
4. For span [i,j], split at each k: check if [i,k] contains X and [k+1,j] contains Y, and if X Y → Z is a rule
5. Mark S at [1,n] if the sentence is grammatical

### 17.2 How to Structure Calculation Answers

**For TF-IDF:**
1. State the formula clearly
2. Compute tf (log-normalised) for each term in each relevant document
3. Compute df for each term
4. Compute IDF = log₁₀(N/df)
5. Multiply: TF-IDF = tf × idf
6. State the final vector and conclusion

**For bigram probability:**
1. Write `P(w₁,w₂,...) = P(w₁|<s>) × P(w₂|w₁) × ...`
2. Look up each bigram count and the denominator unigram count
3. Apply MLE formula: count(w_{i-1},w_i) / count(w_{i-1})
4. Apply Laplace if asked: add 1 to numerator, add V to denominator
5. Multiply all probabilities (or add logs)

**For precision/recall/F1:**
1. Build the confusion matrix explicitly (TP, FP, FN, TN)
2. State formulas
3. Show arithmetic clearly
4. Interpret results in context

### 17.3 Model Answers for Common Question Types

**"Explain X" (2–4 marks):**
- One sentence definition
- Explain the key mechanism
- One concrete example
- One real-world application or implication

**"Compare X and Y" (4–6 marks):**
- Brief definition of each
- Key similarities (1–2)
- Key differences (3–4) — use a table if time allows
- When to use each
- Concrete example demonstrating the difference

**"Describe the process of X" (5–8 marks):**
- Input and output
- Step-by-step process
- Example running through the steps
- Key challenges or limitations

---

## 18. Most Common Exam Questions

### Q1: Zipf's Law (MCQ)
**Q:** Which of the following are true of Zipf's law?
**A:**
- ✓ Word rank and word frequency are inversely related
- ✗ Word rank and frequency are positively correlated
- ✓ It applies to many naturalistic phenomena
- ✓ It describes a power law relationship between rank and frequency

---

### Q2: Perplexity of random digits
**Q:** What is the perplexity of a string of random digits?
**A:** **10** — There are 10 equally likely digits (0–9), so the model predicts each with probability 1/10, giving PP = (1/10)^{-1} = 10.

---

### Q3: Stemming vs. Lemmatisation (MCQ)
**Q:** How does lemmatisation differ from stemming?
**A:**
- ✓ Lemmatisation is informed by linguistic context (POS)
- ✓ Stemming is a more crude, heuristic process
- ✗ Stemming only works for regular verbs (false — it works on all words heuristically)
- ✗ Stemming requires a lexical database (false — stemming is rule-based)

---

### Q4: One-hot encoding shortcomings (MCQ)
**Q:** Shortcomings of one-hot encodings?
**A:**
- ✓ They tend to be relatively sparse
- ✓ They tend to contain many zero elements
- ✗ They tend to be very short (false — they are very LONG = |V|)
- ✗ They tend to be very dense (false — they are sparse)

---

### Q5: N-gram probability calculation
**Q:** In a corpus with counts: my=883,614; beliefs=80,891; my beliefs=378; vocabulary=1,234,567. Estimate P(beliefs|my) with Laplace smoothing.

**A:** P_L(beliefs|my) = (378+1)/(883,614+1,234,567) = **379/2,118,181**

---

### Q6: CFG and ambiguity
**Q:** Explain the ambiguity in "They are hunting dogs."
**A:** This sentence has two parses:
1. **"They are hunting dogs"** = *dogs that hunt* (hunting is an adjective modifying dogs; they are a type of dog)
2. **"They are hunting dogs"** = *they are in the act of hunting dogs* (are+hunting as progressive verb phrase, dogs as object)

Parse tree 1 (they = hunting dogs, a kind of dog):
```
        S
       / \
      NP   VP
      |   /   \
     They VBP   NP
          |   /    \
         are JJ     NN
              |      |
           hunting  dogs
```

Parse tree 2 (they are hunting dogs as prey):
```
        S
       / \
      NP   VP
      |   /   \
     They VBP  VP
          |   /  \
         are VBG  NP
              |    |
           hunting dogs
```
Type of ambiguity: **syntactic/structural ambiguity** — the same word sequence can be assigned two different parse trees with different meanings.

---

### Q7: Confusion matrix calculation
**Q:** What are accuracy, F1, precision, recall for this matrix?
```
Predicted Values
Positive  Negative  Total
   10        20       30    ← Actual Positive
   30        40       70    ← Actual Negative
   40        60      100
```
**A:**
- TP=10, FN=20, FP=30, TN=40
- Accuracy = (10+40)/100 = **50%**
- Precision = 10/(10+30) = 10/40 = **25%**
- Recall = 10/(10+20) = 10/30 ≈ **33.3%**
- F1 = 2×(0.25×0.333)/(0.25+0.333) = 0.167/0.583 ≈ **28.6%**

---

### Q8: Describe NER BIO tagging
**Q:** Label "United Airlines Holding is an ORG" using BIO.
**A:**
```
United/B-ORG  Airlines/I-ORG  Holding/I-ORG  is/O  an/O
```

---

### Q9: Sentence probability with bigrams
**Q:** Calculate P("I am Sam") using the bigram model trained on {<s>I am Sam</s>, <s>Sam I am</s>, <s>I am green</s>}

**A:**
- P(I|<s>) = 2/3
- P(am|I) = 3/3 = 1
- P(Sam|am) = 1/3
- P(</s>|Sam) = 1/2
- **P("I am Sam") = 2/3 × 1 × 1/3 × 1/2 = 2/18 = 1/9**

---

### Q10: Multi-class precision/recall/F1
Already fully worked in Section 14.2. Know the template:
- Per class: TP is on the diagonal; FP is sum of column excluding diagonal; FN is sum of row excluding diagonal
- Micro: sum all TP, FP, FN; micro-P = sum(TP)/sum(TP+FP)
- Macro: average per-class metrics

---

### Q11: Bayes theorem problem (from 2023-Sep)
**Q:** Lauren has 14+6=20 Product A reviews (14 positive, 6 negative) and 24+16=40 Product B reviews (24 positive, 16 negative). A random review is positive. What is P(Product A | positive)?

**A:** Using Bayes' theorem:

- P(A) = 20/60 = 1/3 (prior: 20 out of 60 total reviews are for A)
- P(B) = 40/60 = 2/3
- P(positive|A) = 14/20 = 0.7
- P(positive|B) = 24/40 = 0.6

$$P(A|\text{positive}) = \frac{P(\text{positive}|A) \cdot P(A)}{P(\text{positive})}$$

$$P(\text{positive}) = P(\text{pos}|A)P(A) + P(\text{pos}|B)P(B) = 0.7 \times \frac{1}{3} + 0.6 \times \frac{2}{3} = \frac{0.7}{3} + \frac{1.2}{3} = \frac{1.9}{3}$$

$$P(A|\text{positive}) = \frac{0.7 \times (1/3)}{1.9/3} = \frac{0.7/3}{1.9/3} = \frac{0.7}{1.9} = \frac{7}{19} \approx 0.368$$

---

### Q12: "At least one token is 'the'" probability (from 2023-Sep)
**Q:** 'the' accounts for 10% of tokens. Randomly sample 7 tokens. P(at least one is 'the')?

**Simplifying assumption:** Tokens are drawn **independently** (Markov/unigram assumption — simplification of reality since word occurrence is not independent, but this makes the calculation tractable).

$$P(\text{at least one 'the'}) = 1 - P(\text{no 'the' in 7 tokens})$$
$$= 1 - (1 - 0.1)^7 = 1 - 0.9^7 = 1 - 0.4783 = \mathbf{0.5217}$$

---

### Q13: CFG for home device commands (2023-Sep, 2024-Sep)
**Q:** Design a CFG for commands like "Play the new U2 album", "Turn off the lights", "Search for pasta recipes", "exit", "Stop the music"

**A:**
```
S        → COMMAND
COMMAND  → VERB NP
COMMAND  → VERB NP PREP NP
COMMAND  → 'exit'
NP       → DET ADJ* NOUN
NP       → DET NOUN
NP       → NOUN
PREP     → 'for' | 'by' | 'from'
DET      → 'the' | 'a' | 'all'
ADJ      → 'new' | 'latest' | 'living room' | 'bedroom'
VERB     → 'play' | 'turn' | 'search' | 'stop' | 'dim' | 'increase' | 'mute' | 'shut down'
NOUN     → 'album' | 'lights' | 'music' | 'thermostat' | 'speakers' | 'systems' | 'recipes'
```

**Accepted:** "Play the new U2 album" → VERB NP; "exit" → COMMAND→'exit'

**Rejected (ungrammatical):** "Play" alone → no NP; "the music" → no VERB

---

## 19. One-Page Cheat Sheet

### Key Formulas

**TF-IDF:**
$$w_{t,d} = (1 + \log_{10} \text{tf}_{t,d}) \times \log_{10}\left(\frac{N}{\text{df}_t}\right)$$

**Cosine similarity:**
$$\cos(\vec{q},\vec{d}) = \frac{\vec{q}\cdot\vec{d}}{|\vec{q}||\vec{d}|}$$

**Bigram MLE:**
$$P(w_i|w_{i-1}) = \frac{C(w_{i-1}w_i)}{C(w_{i-1})}$$

**Laplace bigram:**
$$P_L(w_i|w_{i-1}) = \frac{C(w_{i-1}w_i)+1}{C(w_{i-1})+V}$$

**Add-k bigram:**
$$P_k(w_i|w_{i-1}) = \frac{C(w_{i-1}w_i)+k}{C(w_{i-1})+kV}$$

**Perplexity:**
$$\text{PP}(W) = P(w_1\ldots w_N)^{-1/N}$$

**Precision, Recall, F1:**
$$P=\frac{TP}{TP+FP}, \quad R=\frac{TP}{TP+FN}, \quad F1=\frac{2PR}{P+R}$$

**Accuracy:**
$$\text{Acc}=\frac{TP+TN}{TP+TN+FP+FN}$$

**Naïve Bayes:**
$$\hat{c}=\arg\max_c \log P(c) + \sum_i \log P(w_i|c)$$

$$P_{NB}(w|c) = \frac{C(w,c)+1}{\sum_{w'}C(w',c)+|V|}$$

**Zipf's Law:**
$$f(r) \propto \frac{1}{r}$$

**BLEU:**
$$\text{BLEU} = \text{BP} \times \exp\left(\frac{1}{4}\sum_{n=1}^4 \log p_n\right)$$

**Cohen's Kappa:**
$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

### Algorithm Properties Table

| Algorithm | Type | Complexity | Output | Requires |
|-----------|------|-----------|--------|----------|
| Viterbi | DP | O(T²N) | Best tag sequence | HMM (A, B, π) |
| CYK | DP | O(N³|G|) | All parses | CFG in CNF |
| Porter Stemmer | Rule-based | O(L) | Stem | Rules |
| Naïve Bayes | Statistical | O(ND) | Class label | Training data |
| Word2Vec | Neural | O(vocab×dim) | Embeddings | Large corpus |

T = number of tags, N = sentence length, L = word length, D = vocab size

### Tagset Quick Reference

```
NN/NNS/NNP/NNPS — nouns (singular/plural/proper singular/proper plural)
VB/VBD/VBG/VBN/VBP/VBZ — verbs (base/past/gerund/pastpart/nonthird/third)
JJ/JJR/JJS — adjectives (base/comparative/superlative)
RB/RBR/RBS — adverbs
DT — determiner     IN — preposition    CC — coordinating conj
PRP/PRP$ — pronoun  MD — modal          CD — cardinal number
```

### Morphology Quick Reference

```
Inflection: same POS, same lexeme (walk → walks/walked/walking)
Derivation: often changes POS, new lexeme (drive → driver, central → centralize)
Compounding: combining free morphemes (girl + friend = girlfriend)
```

### Smoothing Decision Guide

```
Need simplicity?        → Laplace (add-1)
Need less aggressive?   → Add-k (k<1)
Need principled stats?  → Good-Turing
Need practical backoff? → Katz Backoff
Need best performance?  → Kneser-Ney
Need mixture of all?    → Interpolation
```

### NLTK Quick Reference

```python
# Tokenise
from nltk.tokenize import word_tokenize, sent_tokenize
# POS tag
from nltk import pos_tag
# Stem/Lemmatise
from nltk.stem import PorterStemmer, WordNetLemmatizer
# Stopwords
from nltk.corpus import stopwords
# Corpus
from nltk.corpus import brown, reuters, gutenberg
# Frequency
from nltk import FreqDist, ConditionalFreqDist
# N-grams
from nltk import bigrams, trigrams, ngrams
# NER
from nltk.chunk import ne_chunk
# Language model
from nltk.lm import MLE, Laplace
```

### spaCy Quick Reference

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
doc.ents          # named entities (text, label_)
doc.sents         # sentences
doc.noun_chunks   # base noun phrases
token.pos_        # coarse POS
token.tag_        # fine-grained POS (Penn Treebank)
token.lemma_      # lemma
token.dep_        # dependency relation
token.head        # syntactic head
token.is_stop     # is stop word?
```

### Common Exam Mistakes to Avoid

1. **Forgetting to include `<s>` and `</s>` tokens** in bigram calculations
2. **Using raw TF instead of log-normalised TF** — always use 1+log₁₀(tf) unless told otherwise
3. **Computing IDF as log₁₀(df/N) instead of log₁₀(N/df)** — IDF should be large for rare terms
4. **Confusing micro and macro averaging** — micro = pool all TP/FP/FN; macro = average per-class metrics
5. **Forgetting Brevity Penalty in BLEU** — BP=1 if hypothesis ≥ reference length, else penalty applies
6. **Treating Viterbi backpointers** — always record the previous state (tag), not just the maximum value
7. **In CYK, forgetting to check all split points** — for span [i,j], try every k from i to j-1
8. **Inflection vs derivation** — inflection never changes POS; derivation often does
9. **Word2Vec architectures** — CBOW predicts target from context; Skip-gram predicts context from target (opposite)
10. **Perplexity ordering** — trigram PP < bigram PP < unigram PP (more context = lower perplexity)

---

*End of CM3060 Natural Language Processing – Complete Exam Preparation Guide*

*Primary references: Bird, Klein & Loper (NLTK Book); Jurafsky & Martin (SLP3); Manning, Raghavan & Schütze (IR Book). All worked examples derived from or consistent with past examination papers 2021–2024.*
