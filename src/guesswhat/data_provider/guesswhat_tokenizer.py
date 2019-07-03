from nltk.tokenize import TweetTokenizer
import json

class GWTokenizer:
    """ """
    def __init__(self, dictionary_file,question=True):
        with open(dictionary_file, 'r') as f:
            self.word2i = json.load(f)['word2i']


        self.wpt = TweetTokenizer(preserve_case=False)

        if "<stop_dialogue>" not in self.word2i:
            self.word2i["<stop_dialogue>"] = [len(self.word2i)," "," "]

        self.i2word = {}
        for (k, v) in self.word2i.items():
            # print(k,v)
            self.i2word[v[0]] = k


        # Retrieve key values
        self.no_words = len(self.word2i)
        
        if question:
              self.stop_token = self.word2i["?"][0]

        self.padding_token = self.word2i["<unk>"]
        self.unknown_question_token = self.word2i["<unk>"]

        self.stop_dialogue = self.word2i["<stop_dialogue>"][0]
        self.yes_token = self.word2i["<yes>"][0]
        self.no_token = self.word2i["<no>"][0]
        self.non_applicable_token = self.word2i["<n/a>"][0]
        self.answers = [self.yes_token, self.no_token, self.non_applicable_token]

    """
    Input: String
    Output: List of tokens
    """
    def apply(self, question, is_answer=False,tokent_int = True):

        tokens = []
        if is_answer:
            token = '<' + question.lower() + '>'
            if tokent_int:
                tokens.append(self.word2i[token][0])
            else:tokens.append(token)
        else:
            for token in self.wpt.tokenize(question):
                if token not in self.word2i:
                    token = '<unk>'
                if tokent_int:
                    tokens.append(self.word2i[token][0])
                else:tokens.append(token)

        return tokens

    def decode(self, tokens):
        return ' '.join([self.i2word[tok] for tok in tokens])

    def split_questions(self, dialogue_tokens):

        qas = []
        qa = []
        for token in dialogue_tokens:

            assert token != self.padding_token, "Unexpected padding token"

            # check for end of dialogues
            if token == self.stop_dialogue:
                break

            if token == self.start_token:
                continue

            qa.append(token)

            # check for end of question
            if token in self.answers:
                qas.append(qa)
                qa = []

        return qas



