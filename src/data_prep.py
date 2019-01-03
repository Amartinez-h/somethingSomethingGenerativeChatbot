import re

# Part 1 - Data Preprocessing #
# First cleaning step
def clean_text(text):
    text = text.lower()
    
    # Text replacement
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
                     
    return text    

def data_prep(question_threshold = 20, answer_threshold = 20, length_threshold=25):
# Dataset import
    lines = open('../dataset/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
    conversations = open('../dataset/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')
    
    # Creating a dictionary that matches id and line
    lines_dic = {}
    
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            lines_dic[_line[0]] = _line[4]
            
    # Creating a list of all conversations
    conversations_ids = []
    for conversation in conversations[:-1]:
        _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "").split(",")
        conversations_ids.append(_conversation)
        
    # Getting separately the question and the answer
    questions = []
    answers = []
    
    for conversation in conversations_ids:
        for i in range(0, len(conversation)-1):
            questions.append(lines_dic[conversation[i]])
            answers.append(lines_dic[conversation[i+1]])
    
    # Text cleaning
    clean_questions = []
    clean_answers = []
    
    for question in questions:
        clean_questions.append(clean_text(question))
        
    for answer in answers:
        clean_answers.append(clean_text(answer))
        
    # Creating a dictionary that maps each word with it's number of appaerances
    word_count_dic ={}
    
    for question in clean_questions:
        for word in question.split():
            if word not in word_count_dic:
                word_count_dic[word] = 1
            else:
                word_count_dic[word] += 1
    
    for answer in clean_answers:
        for word in question.split():
            if word not in word_count_dic:
                word_count_dic[word] = 1
            else:
                word_count_dic[word] += 1
                
    # Creating a dictionary to map the question words and answer words to a unique integer
    question_words_dic = {}
    answer_words_dic = {}
    
    n_word = 0
    for word, count in word_count_dic.items():
        if count >= question_threshold:
            question_words_dic[word] = n_word
            n_word += 1
            
    
    n_word = 0
    for word, count in word_count_dic.items():
        if count >= answer_threshold:
            answer_words_dic[word] = n_word
            n_word += 1
        
    # Adding the last tokens to the dictionaries
    tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
    
    for token in tokens:
        question_words_dic[token] = len(question_words_dic)
        
    for token in tokens:
        answer_words_dic[token] = len(question_words_dic)
        
    # Creating the inverse dictionary of answer_words_dic to map the ouput
    answer_dic_words = {w_i: w for w, w_i in answer_words_dic.items()}
    
    # Adding end of line to the end of each answer
    for i in range(len(clean_answers)):
        clean_answers[i] += ' <EOS>'
        
    # Translating all the questions and the answers into integers, and replacing outlayer words by <OUT>
    translated_questions = []
    translated_answers = []
    
    for question in clean_questions:
        translation = []
        for word in question.split():
           if word not in question_words_dic:
               translation.append(question_words_dic['<OUT>'])
           else:
               translation.append(question_words_dic[word])
        translated_questions.append(translation)
        
    for answer in clean_answers:
        translation = []
        for word in answer.split():
           if word not in answer_words_dic:
               translation.append(answer_words_dic['<OUT>'])
           else:
               translation.append(answer_words_dic[word])
        translated_answers.append(translation)
        
    # Sorting the questions and answers by the lenght of the questions
    sorted_clean_questions = []
    sorted_clean_answers = []
    
    for length in range(1, length_threshold + 1):
        for index in enumerate(translated_questions):
            if len(index[1]) == length:
                sorted_clean_questions.append(translated_questions[index[0]])
                
    for length in range(1, length_threshold + 1):
        for index in enumerate(translated_answers):
            if len(index[1]) == length:
                sorted_clean_answers.append(translated_answers[index[0]])
                
        
    return sorted_clean_questions, sorted_clean_answers, answer_dic_words           
               