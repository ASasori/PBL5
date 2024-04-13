import random
scrambled_sentences = []
diction = {}
diction1 = {}
sentences = ["he 123", "934 duie"]
for sentence in sentences:
    # print(scrambled_sentences)
    # print(sentence)
    diction1["source"] = sentence
    diction1["scrambled"] = []
    # Split the sentence into words
    words = sentence.strip().split()
    # print(words)
    # Shuffle the words
    for i in range(0, 3):
        random.shuffle(words)
        # Join the shuffled words back into a sentence
        scrambled_sentence = " ".join(words)
        # Append the scrambled sentence to the list
        diction1["scrambled"].append(scrambled_sentence)
    print("before ", scrambled_sentences)
    diction = diction1
    scrambled_sentences.append(diction)
    print("after ", scrambled_sentences)
