import json
import tensorflow as tf
import numpy as np
import random

def EmbedWord(length_of_word, dim_vector_embedding=64):
    dimension = dim_vector_embedding
    j = np.arange(0, dimension)
    i = np.arange(1, length_of_word + 1)
    b1 = np.random.randint(10, size=(len(i), len(j)))  # Example random array b1
    print("i shape ", i.shape)
    print("j shape ", j.shape)
    # Create masks for odd and even elements of j
    odd_mask = j % 2 == 1
    print(odd_mask.shape)
    # Create masks for broadcasting
    odd_i_mask = np.expand_dims(odd_mask, axis=0)
    j_mask = np.expand_dims(j, axis=0)
    print(odd_i_mask.shape)
    print(j_mask.shape)
    # Add i + j for odd elements of j and i - j for even elements of j
    print(i[:, np.newaxis])
    result = np.where(
        odd_i_mask,
        np.cos(i[:, np.newaxis] * np.power(1 / 10000, (j_mask - 1) / dimension)),
        np.sin(i[:, np.newaxis] * np.power(1 / 10000, j_mask / dimension)),
    )
    return result


def Load_Vocab(url = "output.json"):
    with open(url, "r", encoding="utf-8") as file:
        data = json.load(url)
    list_words = [data1["text"] for data1 in data]
    return list_words


def Tokenize(text):
    list_word = Load_Vocab()
    text = text.lower()
    token = []
    split_text = text.split()
    i = 0
    check_temp = False
    while i < len(split_text):
        j = i
        temp = split_text[j]
        while temp in list_word:
            if j < len(split_text) - 1:
                j += 1
                temp = temp + " " + split_text[j]
            else:
                break
        if (j < len(split_text) - 1):
            token.append(" ".join(temp.split()[:-1]))
            i = j
        else:
            if temp in list_word:
                i = j + 1
                token.append(temp)
            else:
                token.append(" ".join(temp.split()[:-1]))
                i = j
    return token


def Load_Dataset(url = "Dataset.json"):
    with open(url, "r", encoding="utf-8") as file:
        data = json.load(file)


def scramble_sentences(sentences):
    scrambled_sentences = []
    diction = {}
    for sentence in sentences:
        diction["source"] = sentence
        diction["scrambled"] = []
        # Split the sentence into words
        words = sentence.strip().split()
        # Shuffle the words
        for i in range(0, 3):
            random.shuffle(words)
            # Join the shuffled words back into a sentence
            scrambled_sentence = " ".join(words)
            # Append the scrambled sentence to the list
            diction["scrambled"].append(scrambled_sentence)
        scrambled_sentences.append(diction)
    return scrambled_sentences


