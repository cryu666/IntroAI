import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import WordNetLemmatizer


def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)
    
    # TO-DO 0: Other preprocessing function attemption
    # Begin your code 
    
    preprocessed_text = re.sub("<[^>]+>", " ", preprocessed_text).strip()
    
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(preprocessed_text)
    tokens = [token.strip() for token in tokens]
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    preprocessed_text = ' '.join(lemmatized_tokens)

    # End your code

    return preprocessed_text