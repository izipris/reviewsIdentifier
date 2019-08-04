import nltk
from utilities.contractions import contractions_dict
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pytypo
stopword = stopwords.words('english')


class LanguageUtils:
    """see https://medium.com/@pemagrg/pre-processing-text-in-python-ad13ea544dae"""
    @staticmethod
    def to_lower(text):
        """
        Converting text to lower case
        """
        return text.lower()

    @staticmethod
    def spelling_correction(text):
        """basic spelling correction"""
        return pytypo.correct_sentence(text)

    @staticmethod
    def remove_contractions(text):
        """strips contractions (ain't-am not, can't-cannot,...)"""
        tokenized = text.split(' ')
        final = []
        for word in tokenized:
            final.append(contractions_dict[word] if word in contractions_dict else word)
        return ' '.join(final)

    @staticmethod
    def remove_numbers(text):
        """strips numbers"""
        return ''.join(c for c in text if not c.isdigit())

    @staticmethod
    def remove_punctuation(s):
        """strips punctuation"""
        return ''.join(c for c in s if c not in punctuation)

    @staticmethod
    def remove_stop_words(text):
        """strips stop words (a, is, ...)"""
        word_tokens = nltk.word_tokenize(text)
        removing_stopwords = [word for word in word_tokens if word not in stopword]
        return ' '.join(removing_stopwords)

    @staticmethod
    def lemmatize_sentence(text):
        """Migrate words to their basic - e.g functionality to function"""
        wordnet_lemmatizer = WordNetLemmatizer()
        word_tokens = nltk.word_tokenize(text)
        lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in word_tokens]
        return ' '.join(lemmatized_word)

