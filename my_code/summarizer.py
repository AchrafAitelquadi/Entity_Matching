import os
from collections import Counter
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import get_tokenizer
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
class Summarizer:
    """To summarize a data entry pair into length up to the max sequence length.

    Args:
        task_config (Dictionary): the task configuration
        lm (string): the language model (bert, albert, or distilbert)

    Attributes:
        config (Dictionary): the task configuration
        tokenizer (Tokenizer): a tokenizer from the huggingface library
    """

    def __init__(self, task_config, lm, inference=False, inference_file=None):
        self.config = task_config
        self.tokenizer = get_tokenizer(lm)
        self.len_cache = {}
        self.inference = inference
        self.inference_file = inference_file
        self.vocab = None
        self.idf = None

        # Always build index
        self.build_index()
        
    def build_index(self):
        """Build the idf index depending on mode (train or inference)."""
        content = []

        if not self.inference:
            # Training mode → use train/valid/test sets
            fns = [self.config["trainset"],
                   self.config["validset"],
                   self.config["testset"]]
        else:
            # Inference mode → require a blocked pairs file
            if not self.inference_file:
                raise ValueError("Inference mode requires inference_file path")
            fns = [self.inference_file]

        # Collect content
        for fn in fns:
            with open(fn) as fin:
                for line in fin:
                    LL = line.strip().split('\t')
                    # include all except last if training, include all if inference
                    if not self.inference:
                        if len(LL) > 2:
                            for entry in LL:
                                content.append(entry)
                    else:
                        # include all except last if it's a label? or include all?
                        for entry in LL[:-1]:  # exclude label for TF-IDF
                            content.append(entry)

        # Fit TF-IDF
        vectorizer = TfidfVectorizer().fit(content)
        self.vocab = vectorizer.vocabulary_
        self.idf = vectorizer.idf_
    
    def get_len(self, word):
        """
        Return the sentence_piece length of a token.
        """
        if word in self.len_cache:
            return self.len_cache[word]
        length = len(self.tokenizer.tokenize(word))
        self.len_cache[word] = length
        return length
    
    def transform(self, row, max_len=128):
        """
        Summarize one single example.

        Only retain tokens of the highest tf-idf.

        Args:
            row (str): a matching example of two data entries and a binary label, separated by tab
            max_len (int, Optional): the maximum sequence length to be summerized to
        
        Returns:
            str: the summarized example
        """
        parts = row.strip().split('\t')

        if self.inference:
            if len(parts) < 2:
                return ""
            sentA, sentB = parts[:2]
            label = parts[2] if len(parts) > 2 else None 
        else:
            if len(parts) < 3:
                return ""
            sentA, sentB, label = parts[:3]
            
        res=''
        cnt = Counter()
        for sent in [sentA, sentB]:
            tokens = sent.split(' ')
            for token in tokens:
                if token not in ['COL', 'VAL'] and token not in stopwords:
                    if token in self.vocab:
                        cnt[token] += self.idf[self.vocab[token]]
        

        for sent in [sentA, sentB]:
            token_cnt = Counter(sent.split(' '))
            total_len = token_cnt['VAL'] + token_cnt['COL']

            subset = Counter()
            for token in set(token_cnt.keys()):
                subset[token] = cnt[token]
            subset = subset.most_common(max_len)

            topk_tokens_copy = set([])
            for word, _ in subset:
                bert_len = self.get_len(word)
                if bert_len + total_len > max_len:
                    break
                total_len += bert_len
                topk_tokens_copy.add(word)
            
            for token in sent.split(' '):
                if token in ['COL', 'VAL']:
                    res += token + ' '
                elif token in topk_tokens_copy:
                    res += token + ' '
                    topk_tokens_copy.remove(token)

            res += '\t'

        if label is not None:
            res += label + '\n'
        else:
            res = res.rstrip('\t ') + '\n'

        return res
    
    def transform_file(self, input_fn, max_len = 256, overwrite = False):
        """Summarize all lines of a tsv file.

        Run the summarizer. If the output already exists, just return the file name.

        Args:
            input_fn (str): the input file name
            max_len (int, optional): the max sequence len
            overwrite (bool, optional): if true, then overwrite any cached output

        Returns:
            str: the output file name
        """
        out_fn = input_fn.replace("input", "working") + '.su'
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        if not os.path.exists(out_fn) or os.stat(out_fn).st_size == 0 or overwrite:
            with open(out_fn, "w") as fout:
                for line in open(input_fn):
                    fout.write(self.transform(line, max_len = max_len))
        
        return out_fn