from collections import defaultdict
from . import __version__
import numpy as np
import random
import re

class Tokenizer:
    """Class for tokenization

    TBI:
        Leminization
        Custom tokens
        Stemming
        Languages
        Stop words"""

    def __init__(
            self,
            max_length: int = None,
            lowercase: bool = True,
            punctuation: bool = False,
            pad_token: str = "<PAD>",
            unknown_token: str = "<UNK>"
        ) -> None:

        self.punctuation = punctuation  # NYI
        self.max_length = max_length
        self.pad_token = pad_token
        self.lowercase = lowercase
        self._tokens = {pad_token: 0, unknown_token: 1}
        self._index_to_token = {0: pad_token, 1: unknown_token}  # Reverse mapping for decoding
        self.unknown_token = unknown_token

    def encode(self, text: str) -> list[int]:
        # Remove punctuation (if applicable) and split into tokens
        if not self.punctuation:
            tokens = re.sub(r"[^\w\s]+", "", text)

        tokens = tokens.split()

        # Convert to lowercase if specified
        if self.lowercase:
            tokens = [token.lower() for token in tokens]

        # Cap the length of the tokens list if max_length is specified
        if self.max_length is not None:
            tokens = tokens[:self.max_length]
            while len(tokens) < self.max_length:
                tokens.append(self.pad_token)

        encoded_tokens = []

        for token in tokens:
            if token in self._tokens:
                # Append the existing token index to the encoded tokens list
                encoded_tokens.append(self._tokens[token])
            else:
                # Assign a new index to the new token and append it
                index = len(self._tokens)
                self._tokens[token] = index
                self._index_to_token[index] = token  # Update reverse mapping
                encoded_tokens.append(index)

        return encoded_tokens  # Return the list of encoded token indices
    
    def fetch_token(self, word: str) -> int:
        # if word is in tokens, return its index
        if not self.punctuation:
            word = re.sub(r"[^\w\s]+", "", word)

        if self.lowercase:
            word = word.lower()

        if word in self._tokens:
            return self._tokens[word]

        else:
            # if word is unknown, assign a new index, add it to tokens, and return its index
            index = len(self._tokens)
            self._tokens[word] = index
            self._index_to_token[index] = word
            return index

    def fetch_word(self, token: int) -> str:
        return self._index_to_token.get(token, self.unknown_token)

    def decode(self, encoded: list[int]) -> str:
        decoded = []

        for token_index in encoded:
            if token_index in self._index_to_token:
                decoded.append(self._index_to_token[token_index])
            else:
                decoded.append(self.unknown_token)  # Handle unknown tokens gracefully
        
        return " ".join(decoded)  # Return the decoded sentence as a string

class MarkovChain:
    """Class for handling creation and manipulation of Markov Chains, rough implementation, optimization TBI"""
    def __init__(self, text: str, order: int = 1) -> None:
        self.order = order
        self.model = defaultdict(list)
        self._build_model(text)

    def _build_model(self, text: str) -> None:
        words = text.split()
        for i in range(len(words) - self.order):
            # Get the current state (the current order of words)
            state = tuple(words[i:i + self.order])
            next_word = words[i + self.order]
            self.model[state].append(next_word)

    def generate(self, length: int = 50) -> str:
        # Start from a random state
        current_state = random.choice(list(self.model.keys()))
        output_words = list(current_state)

        for _ in range(length):
            next_words = self.model.get(current_state, None)
            
            if not next_words: break  # No next words found; end generation

            next_word = random.choice(next_words)
            output_words.append(next_word)
            
            # Update current state
            current_state = tuple(output_words[-self.order:])  # Keep only the last 'order' words

        return " ".join(output_words)

class English:
    """Class for handling typical English characters (and some common characters), not reccommended for any chatbot style neural network, adding only by request"""

    def __init__(self) -> None:
        self.chars = {
            " ": 0,  "a": 1,  "b": 2,  "c": 3,  "d": 4,  "e": 5,  "f": 6,  "g": 7,  "h": 8,  "i": 9,
            "j": 10, "k": 11, "l": 12, "m": 13, "n": 14, "o": 15, "p": 16, "q": 17, "r": 18, "s": 19,
            "t": 20, "u": 21, "v": 22, "w": 23, "x": 24, "y": 25, "z": 26,
            
            "A": 27, "B": 28, "C": 29, "D": 30, "E": 31, "F": 32, "G": 33, "H": 34, "I": 35, "J": 36,
            "K": 37, "L": 38, "M": 39, "N": 40, "O": 41, "P": 42, "Q": 43, "R": 44, "S": 45, "T": 46,
            "U": 47, "V": 48, "W": 49, "X": 50, "Y": 51, "Z": 52,

            "0": 53, "1": 54, "2": 55, "3": 56, "4": 57, "5": 58, "6": 59, "7": 60, "8": 61, "9": 62,

            "`": 63, "!": 64, '"': 65, "£": 66, "$": 67, "%": 68, "^": 69, "&": 70, "*": 71, "(": 72,
            ")": 73, "-": 74, "_": 75, "=": 76, "+": 77, "/": 78, "|": 79, "[": 80, "]": 81, "{": 82,
            "}": 83, ";": 84, ":": 85, "'": 86, "@": 87, "#": 88, "~": 89, ",": 90, "<": 91, ".": 92,
            ">": 93, "?": 94, "¡": 95, "¿": 96, "“": 97, "”": 98, "€": 99,
        }

        self.max_length = 100

    def encode(self, text: str) -> list[int]:
        encoded_list = [self.chars[char] for char in text if char in self.chars]
        return np.pad(encoded_list, (0, self.max_length - len(encoded_list)), "constant")[:self.max_length]

    def decode(self, encoded_list: list[int]) -> str:
        reverse_chars = {value: key for key, value in self.chars.items()}
        decoded_list = [reverse_chars[encoded_char] for encoded_char in encoded_list if encoded_char in reverse_chars]
        return "".join(decoded_list)

class BagOfWords:
    def __init__(self, corpus: list[str] = None, binary: bool = False) -> None:
        self.corpus = corpus if corpus is not None else []
        self.binary = binary
        self.vocab = self._build_vocab(self.corpus)

    def _build_vocab(self, corpus: list[str]) -> dict:
        # Initialize an empty set to collect unique words
        vocab_set = set()

        # Tokenize each document in the corpus and add to the vocab set
        for doc in corpus:
            tokens = self._tokenize(doc)
            vocab_set.update(tokens)

        # Map each word to a unique index
        return {word: idx for idx, word in enumerate(sorted(vocab_set))}

    def _tokenize(self, text: str, ignore_case: bool = True, ignore_punctuation: bool = True) -> list[str]:
        tokenized_str = text

        if ignore_case:
            tokenized_str = tokenized_str.lower()

        if ignore_punctuation:
            tokenized_str_no_punctuation = []
            for char in tokenized_str:
                if char.isalpha() or char == " ":
                    tokenized_str_no_punctuation.append(char)
            tokenized_str = "".join(tokenized_str_no_punctuation)

        return tokenized_str.split()

    def convert(self, sentence: str) -> np.ndarray:
        # Initialize a zero vector for the BoW representation
        bow_vector = np.zeros(len(self.vocab), dtype=int)

        # Tokenize the input sentence
        tokens = self._tokenize(sentence)

        # For each token, increment the corresponding index in the BoW vector
        for token in tokens:
            if token in self.vocab:  # Only count words that are in the vocab
                index = self.vocab[token]
                if self.binary: bow_vector[index] = 1
                else: bow_vector[index] += 1

        return bow_vector

class CharacterPredictor:
    def __init__(self, text: str = None) -> None:
        self.relationships = {}
        text_list = list(text)

        # Iterate until the penultimate character
        for index in range(len(text_list) - 1):
            char = text_list[index]
            next_char = text_list[index + 1]

            # Add the next character to the list for this character
            if char in self.relationships:
                self.relationships[char].append(next_char)

            else:
                self.relationships[char] = [next_char]

    def predict(self, char: str, method: str = "random") -> str:
        """
        Predict the next character based on the given character.

        Args:
            char (str): The character for which to predict the next character.
            method (str): The method of prediction (default is 'random').

        Returns:
            str or None: The predicted character or None if no prediction can be made.

        Raises:
            ValueError: If the specified method is unsupported.
        """

        method = method.lower().strip()

        if char not in self.relationships:
            return None # Stays silent to reduce errors

        elif method == "random":
            return random.choice(self.relationships[char])

        elif method == "frequency" or method == "frequency-random" or method == "frequency-first":
            # Find the next character with the highest frequency
            frequencies = {}
            for character in self.relationships[char]:
                if character in frequencies: frequencies[character] += 1
                else: frequencies[character] = 1                 

            # Find the maximum frequency
            max_freq = max(frequencies.values())
            candidates = [char for char, freq in frequencies.items() if freq == max_freq]

            # Choose based on the specified method
            if method == "frequency" or method == "frequency-random":
                return random.choice(candidates)  # Default behavior is random if there's a tie

            elif method == "frequency-first":
                return candidates[0]  # Choose the first one in the list

        else:
            raise ValueError(f"Unsupported prediction method: \"{method}\" as of pybernetics v{__version__}")

class WordPredictor:
    def __init__(self, text: str = None) -> None:
        self.word_relationships = {}
        if text:
            text_list = text.split()

            # Iterate through the text list to build relationships
            for index in range(len(text_list) - 1):
                word = text_list[index]
                next_word = text_list[index + 1]

                # Add the next word to the list for this word
                if word in self.word_relationships:
                    self.word_relationships[word].append(next_word)
                else:
                    self.word_relationships[word] = [next_word]

    def predict(self, word: str, method: str = "random") -> str:
        """
        Predict the next word based on the given word.

        Args:
            word (str): The word for which to predict the next word.
            method (str): The method of prediction (default is 'random').

        Returns:
            str or None: The predicted word or None if no prediction can be made.

        Raises:
            ValueError: If the specified method is unsupported.
        """
        method = method.lower().strip()

        if word not in self.word_relationships:
            return None  # No prediction can be made

        elif method == "random":
            return random.choice(self.word_relationships[word])

        elif method in ["frequency", "frequency-random", "frequency-first"]:
            # Count the frequencies of the next words
            frequencies = {}
            for next_word in self.word_relationships[word]:
                if next_word in frequencies:
                    frequencies[next_word] += 1
                else:
                    frequencies[next_word] = 1

            # Find the maximum frequency
            max_freq = max(frequencies.values())
            candidates = [word for word, freq in frequencies.items() if freq == max_freq]

            # Choose based on the specified method
            if method == "frequency" or method == "frequency-random":
                return random.choice(candidates)  # Default behavior is random if there's a tie

            elif method == "frequency-first":
                return candidates[0]  # Choose the first one in the list

        else:
            raise ValueError(f"Unsupported prediction method: \"{method}\".")
        
class Phrases:
    """
    Class for handling common phrases and their affirmative and negative responses.
    
    EG:
    ```
    from pybernetics.NaturalLanguageProcessing import Phrases

    while True:
        user_input = input("Would you like to continue? ")
        if user_input.lower() in Phrases.affirmative:
            print("Continuing...")
            continue
        elif user_input.lower() in Phrases.negative:
            print("Exiting...")
            exit()
        else:
            print("Please enter a valid response.")
    ```

    Notes:
        - The affirmative and negative lists are case-insensitive.
        - The affirmative list contains common affirmative responses.
        - The negative list contains common negative responses.
        - The lists are not exhaustive and may not cover all possible responses.
        - The lists are not language-specific and may not work for all languages.
        - They are intended for use in simple chatbots and similar applications.
    """
    affirmative = [
        "yes",
        "ofc",
        "of course",
        "sure",
        "why not",
        "y not",
        "y",
        "im down",
        "i'm down",
        "yessir",
        "affirmative",
        "positive",
        "yeah",
        "yes please",
        "fine",
        "i guess",
        "ig",
        "positive",
        "yea",
        "ye",
        "yep",
        "yup",
    ]

    negative = [
        "no",
        "negative",
        "nah",
        "nuh uh",
        "nuh-uh",
        "no way",
        "im good",
        "i'm good",
        "no thanks",
        "no way hoze",
        "ofcource not",
        "definitely not",
        "absolutely not",
        "nope",
        "n",
        "no sir",
        "no ma'am",
        "no maam",
        "no mam",
        "i dont think so",
        "oh no",
    ]
