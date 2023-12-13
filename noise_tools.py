from random import sample
import math

def delete_words(text, numDeletions):
    textArr = text.split(' ')
    sequence = [i for i in range(len(textArr))]
    indices = sample(sequence, numDeletions)
    indices = sorted(indices)
    #print(indices)
    #print(textArr, sequence, indices)
    newText = ""
    deletionIndex = 0
    for i in range(len(textArr)):
        if deletionIndex >= len(indices) or i != indices[deletionIndex]:
            newText += textArr[i] + " "
        else:
            deletionIndex += 1
    return newText

def swap_string(string, inds):
    char_list = list(string)
    char_list[inds[0]], char_list[inds[1]] = char_list[inds[1]], char_list[inds[0]]
    new_string = "".join(char_list)
    return new_string

def swap_characters(text, numSwaps):
    textArr = text.split(' ')
    sequence = [i for i in range(len(textArr))]
    indices = sample(sequence, numSwaps)
    indices = sorted(indices)
    indicesInd = 0
    newText = ""
    for i in range(len(textArr)):
        if (indicesInd < len(indices) and i == indices[indicesInd]):
            index = indices[indicesInd]
            word = textArr[i]
            wordSeq = [i for i in range(len(word))]
            if len(wordSeq) <= 1:
                word = textArr[i]
            elif len(wordSeq) == 2:
                word = swap_string(word, wordSeq)
            else:
                wordIndices = sample(wordSeq, 2)
                word = swap_string(word, wordIndices)
            newText += word + " "
            indicesInd += 1
        else:
            newText += textArr[i] + " "
    return newText

def addCommonWords(text, numNoiseDesired):
    typoFile = open("CommonWords.txt", "r")
    content = typoFile.read()
    arr = content.split('\n')
    #print(arr)
    #newArr = sample(sequence, numTyposDesired)
    newArr = []
    for i in range(len(arr)):
        #print(word)
        word = arr[i]
        if len(word) > 0 and word[0] != '$':
            newArr.append(word)
    #print(newArr)
    sequence = [i for i in range(len(newArr))]
    indices = sample(sequence, numNoiseDesired)
    textArr = text.split(' ')
    sequence = [i for i in range(len(textArr))]
    positionsInText = sample(sequence, numNoiseDesired)
    finalArr = textArr
    posIndex = 0
    for index in indices:
        finalArr.insert(positionsInText[posIndex], newArr[index])
        posIndex += 1
    return " ".join(finalArr)

def noise_algorithm(text):
    #print(text)
    numNoiseData = min(512, math.floor(0.1*len(text.split(' '))))
    text = delete_words(text,numNoiseData)
    #print("Text after deletions :", text)
    text = swap_characters(text, numNoiseData)
    #print("Text after swaps :", text)
    text = addCommonWords(text, numNoiseData)
    #print("Text after added words :", text)
    return text