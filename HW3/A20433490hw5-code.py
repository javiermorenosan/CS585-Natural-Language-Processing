import math, random

################################################################################
# Part 0: Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    text = start_pad(n) + text
    ngrams_list = []
    for i in range(n, len(text)):
        ngram = text[i-n:i]
        ngrams_list.append((ngram, text[i]))
    return ngrams_list

def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.vocab = []
        self.text = ''
        self.ngrams_counts = {}
        self.contexts_counts = {}
        pass

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        self.text = text
        for c in self.text:
            if c not in self.vocab:
                self.vocab.append(c)
        ngrams_list = ngrams(self.n, text)
        for ngram in ngrams_list:
            if ngram[0] in self.contexts_counts:
                self.contexts_counts[ngram[0]] += 1
            else:
                self.contexts_counts[ngram[0]] = 1
            
            if ngram in self.ngrams_counts:
                self.ngrams_counts[ngram] += 1
            else:
                self.ngrams_counts[ngram] = 1
        return

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        ngram = (context, char)
        
        if self.k == 0:
            if context not in self.contexts_counts:
                prob = 1/len(self.vocab)
            else:
                if ngram not in self.ngrams_counts:
                    ngram_count = 0.0
                else:
                    ngram_count = self.ngrams_counts[ngram]
                if ngram[0] not in self.contexts_counts:
                    context_count = 0.0
                else:
                    context_count = self.contexts_counts[ngram[0]]
                if context_count == 0:
                    prob = 0.0
                else:
                    prob = ngram_count/context_count
        else:
            if context not in self.contexts_counts:
                prob = 1/len(self.vocab)
            else:
                if ngram not in self.ngrams_counts:
                    ngram_count = self.k
                else:
                    ngram_count = self.ngrams_counts[ngram]+self.k
                if ngram[0] not in self.contexts_counts:
                    context_count = len(self.vocab)
                else:
                    context_count = self.contexts_counts[ngram[0]] + self.k*len(self.vocab)
                
                if context_count == 0:
                    prob = 0.0
                else:
                    prob = (ngram_count)/(context_count)   
        return prob

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        V = sorted(self.vocab)
        r = random.random()
        character = ''
        for i in range (0, len(V)):
            part1 = 0.0
            part2 = 0.0
            for j in range(0, i):
                part1 += self.prob(context, V[j])
            for j in range(0, i+1):
                part2 += self.prob(context, V[j])
            if r >=part1 and r<part2:
                character = V[i]
            else:
                pass
        return character

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        text = ""
        for i in range(0, self.n):
            text += '~'
        context = text
        for i in range(0, length):
            newChar = self.random_char(context)
            text += newChar
            context = text[len(text)-self.n:]
        
        text = text[self.n:]
        return text

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        completeText=""
        for i in range(0, self.n):
            completeText += '~'
        completeText += text[:len(text)]
        perplexity = 0.0
        N = len(text)
        for i in range(0, N):
            context = completeText[i:i+self.n]
            char = completeText[i+self.n]
            Pi = self.prob(context, char)
            if (Pi == 0.0):
                return float('inf')
            perplexity += -(math.log(Pi))
        perplexity = math.exp((1/N)*(perplexity))
        return perplexity

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.vocab = []
        self.lambdas = []
        self.models = []
        for i in range(0, n+1):
            l=1/(n+1)
            m = NgramModel(i, self.k)
            self.models.append(m)
            self.lambdas.append(l)
        pass

    def get_vocab(self):
        return self.vocab

    def update(self, text):
        for i in range(0, self.n+1):
            self.models[i].update(text)
        return

    def prob(self, context, char):
        prob = 0.0
        for i in range(0, self.n+1):
            probi = self.models[i].prob(context[self.n-i:], char)
            prob += probi*self.lambdas[i]
        return prob
    
    def setLambdas(self, lambdas):
        if sum(lambdas) != 1.0:
            print("The sum of lambdas must be equal to 1.0")
            return
        else:
            self.lambdas = lambdas

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    print("\n###################### PART 1 ######################")
    print("\nModel with 2-grams: ")
    print(" ")
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 2)
    random.seed(1)
    print(m.random_text(250))

    print("\nModel with 3-grams: ")
    print(" ")
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 3)
    random.seed(1)
    print(m.random_text(250))

    print("\nModel with 4-grams: ")
    print(" ")
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 4)
    random.seed(1)
    print(m.random_text(250))

    print("\nModel with 7-grams: ")
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7)
    random.seed(1)
    print(m.random_text(250))
    
    print("\n###################### PART 2: Smoothing ######################")
    print(" ")
    
    #We are going to calculate the average perplexity of the model with each corpus taking 20 characters sentence
    print("\nWith 3-grams models we calculate the average perplexity of the model with both corpus for different values of the smoothing parameter k:")
    
    f1 = open('./test_data/shakespeare_sonnets.txt', encoding='utf-8', errors='ignore')
    sh_txt = f1.read()
    sh_list = []
    sentence = ""
    for i in range(0, len(sh_txt)):
        sentence += sh_txt[i]
        if i != 0 and i%20 == 0:
            sh_list.append(sentence)
            sentence = ""
        
    f2 = open('./test_data/nytimes_article.txt', encoding='utf-8', errors='ignore')
    ny_txt = f2.read()
    ny_list = []
    sentence = ""
    for i in range(0, len(ny_txt)):
        sentence += ny_txt[i]
        if i != 0 and i%20 == 0:
            ny_list.append(sentence)
            sentence = ""
    
    ks =[1, 3, 5, 100]
    for k in ks:
        print("\nk =", k)
        m = create_ngram_model_lines(NgramModel, 'shakespeare_input.txt', 3, k)
        
        perplexities_sh = []
        for s in sh_list:
            perplexities_sh.append(m.perplexity(s))
        avg_perplexity_sh = sum(perplexities_sh)/len(perplexities_sh)
        
        perplexities_ny = []
        for s in ny_list:
            perplexities_ny.append(m.perplexity(s))
        avg_perplexity_ny = sum(perplexities_ny)/len(perplexities_ny)
        
        print("Average perplexity of the model with Shakespeare's sonnets:", avg_perplexity_sh)
        print("Average perplexity of the model with New York Times:", avg_perplexity_ny)
    
    print("\n###################### PART 2: Interpolation ######################")
    print("With 3-grams models and a k = 1 we calculate the average perplexity of the model with both corpus for different values of lambdas:")
    
    lambdas_list = [[0.25, 0.25, 0.25, 0.25], [0.75, 0.25/3.0, 0.25/3.0, 0.25/3.0], [0.25/3.0, 0.75, 0.25/3.0, 0.25/3.0], [0.25/3.0, 0.25/3.0, 0.75, 0.25/3.0], [0.25/3.0, 0.25/3.0, 0.25/3.0, 0.75], [0.01, 0.09, 0.2, 0.7]]
    for lambdas in lambdas_list:
        print("\nLambdas:", lambdas)
        m = create_ngram_model_lines(NgramModelWithInterpolation, 'shakespeare_input.txt', 3, 1)
        m.setLambdas(lambdas)
        
        perplexities_sh = []
        for s in sh_list:
            perplexities_sh.append(m.perplexity(s))
        avg_perplexity_sh = sum(perplexities_sh)/len(perplexities_sh)
        
        perplexities_ny = []
        for s in ny_list:
            perplexities_ny.append(m.perplexity(s))
        avg_perplexity_ny = sum(perplexities_ny)/len(perplexities_ny)
        
        print("Average perplexity of the model with Shakespeare's sonnets:", avg_perplexity_sh)
        print("Average perplexity of the model with New York Times:", avg_perplexity_ny)
    